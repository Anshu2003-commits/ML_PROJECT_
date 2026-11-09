// Build report/REPORT.docx from report/REPORT.md (A4, headings, tables, figures, page numbers).
// Setup once:  npm install docx@9.5.1   (in any folder on NODE_PATH, or here)
// Run from the repo root:  node report/build_docx.js . report/REPORT.docx

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow,
  TableCell, WidthType, BorderStyle, ShadingType, ExternalHyperlink, Footer, PageNumber,
  LevelFormat, TableOfContents,
} = require("docx");

const ROOT = process.argv[2];
const OUT = process.argv[3];
const REPORT_DIR = path.join(ROOT, "report");
const GITHUB = "https://github.com/Anshu2003-commits/ML_PROJECT_/blob/v1.0/";
const FONT = "Calibri";
const CONTENT_DXA = 11906 - 2 * 1440;  // A4 width minus 1" margins
const INK = "1F1F1F", MUTED = "595959", HEAD_FILL = "E8EEF7", BORDER = "BFBFBF";

const md = fs.readFileSync(path.join(REPORT_DIR, "REPORT.md"), "utf8").replace(/\r\n/g, "\n");

// ---------- inline markdown -> runs ----------
function resolveLink(href) {
  if (/^https?:/.test(href)) return href;
  const clean = href.split("#")[0];
  const rel = path.posix.normalize(path.posix.join("report", clean));
  return GITHUB + rel;
}

function inline(text, base = {}) {
  const out = [];
  const re = /(\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))/g;
  let last = 0, m;
  const plain = (t, extra = {}) => t && out.push(new TextRun({ text: t, font: FONT, ...base, ...extra }));
  while ((m = re.exec(text))) {
    plain(text.slice(last, m.index));
    if (m[2] !== undefined) out.push(...inline(m[2], { ...base, bold: true }));
    else if (m[3] !== undefined) out.push(...inline(m[3], { ...base, italics: true }));
    else if (m[4] !== undefined) plain(m[4], { font: "Consolas", size: (base.size || 22) - 2 });
    else if (m[5] !== undefined) {
      out.push(new ExternalHyperlink({
        link: resolveLink(m[6]),
        children: [new TextRun({ text: m[5], font: FONT, ...base, color: "1F5FBF", underline: {} })],
      }));
    }
    last = re.lastIndex;
  }
  plain(text.slice(last));
  return out;
}

// ---------- block builders ----------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

function image(rel) {
  const file = path.join(REPORT_DIR, rel);
  const { w, h, data } = pngSize(file);
  const maxW = 600, maxH = 430;            // px at 96 dpi: 6.25 in wide (text width 6.27 in), 4.5 in tall
  const scale = Math.min(maxW / w, maxH / h, 1);
  const width = Math.round(w * scale), height = Math.round(h * scale);
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new ImageRun({ type: "png", data, transformation: { width, height },
      altText: { title: path.basename(rel), description: path.basename(rel), name: path.basename(rel) } })],
  });
}

function table(lines) {
  const rows = lines.filter((l, i) => i !== 1).map(l =>
    l.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
  const n = rows[0].length;
  const lens = Array.from({ length: n }, (_, j) =>
    Math.max(...rows.map(r => (r[j] || "").replace(/\*\*|`/g, "").length), 3));
  const capped = lens.map(x => Math.min(x, 40));
  const total = capped.reduce((a, b) => a + b, 0);
  const widths = capped.map(x => Math.floor(CONTENT_DXA * x / total));
  widths[n - 1] += CONTENT_DXA - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: BORDER };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: CONTENT_DXA, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      cantSplit: true,
      children: r.map((cell, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        borders,
        shading: i === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: HEAD_FILL } : undefined,
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        children: [new Paragraph({
          spacing: { before: 0, after: 0 },
          children: inline(cell, { size: 18, bold: i === 0 ? true : undefined, color: INK }),
        })],
      })),
    })),
  });
}

function para(text, opts = {}) {
  return new Paragraph({ spacing: { before: 0, after: 120, line: 276 }, ...opts, children: inline(text, opts.run || {}) });
}

// ---------- parse ----------
const children = [];
const lines = md.split("\n");
let i = 0;
let section = "";
while (i < lines.length) {
  const line = lines[i];
  const t = line.trim();
  if (!t) { i++; continue; }

  if (t.startsWith("```")) {                         // code block
    const code = [];
    i++;
    while (i < lines.length && !lines[i].trim().startsWith("```")) code.push(lines[i++]);
    i++;
    code.forEach((c, k) => children.push(new Paragraph({
      spacing: { before: k === 0 ? 60 : 0, after: k === code.length - 1 ? 120 : 0 },
      shading: { type: ShadingType.CLEAR, color: "auto", fill: "F3F3F3" },
      children: [new TextRun({ text: c || " ", font: "Consolas", size: 16 })],
    })));
    continue;
  }
  if (t === "---") {                                  // horizontal rule
    children.push(new Paragraph({ spacing: { before: 60, after: 120 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BORDER, space: 1 } }, children: [] }));
    i++; continue;
  }
  if (t.startsWith("# ")) {
    children.push(new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, children: inline(t.slice(2)) }));
    i++; continue;
  }
  if (t.startsWith("## ")) {
    section = t.slice(3);
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: inline(t.slice(3)) }));
    i++; continue;
  }
  if (t.startsWith("### ")) {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: inline(t.slice(4)) }));
    i++; continue;
  }
  const img = t.match(/^!\[[^\]]*\]\(([^)]+)\)$/);
  if (img) { children.push(image(img[1])); i++; continue; }
  if (t.startsWith("|")) {                            // table
    const tl = [];
    while (i < lines.length && lines[i].trim().startsWith("|")) tl.push(lines[i++]);
    children.push(table(tl));
    children.push(new Paragraph({ spacing: { before: 0, after: 120 }, children: [] }));
    continue;
  }
  if (t.startsWith("- ")) {                           // bullet list (references: hanging indent instead)
    const isRef = section === "References";
    while (i < lines.length && lines[i].trim().startsWith("- ")) {
      const content = inline(lines[i].trim().slice(2), isRef ? { size: 20 } : {});
      children.push(isRef
        ? new Paragraph({ indent: { left: 540, hanging: 540 }, spacing: { before: 0, after: 100 }, children: content })
        : new Paragraph({ numbering: { reference: "bullets", level: 0 },
            spacing: { before: 0, after: 80, line: 276 }, children: content }));
      i++;
    }
    continue;
  }
  if (/^\*Figure \d/.test(t)) {                       // figure caption
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 200 },
      children: inline(t.replace(/^\*|\*$/g, ""), { italics: true, size: 18, color: MUTED }) }));
    i++; continue;
  }
  // paragraph: join consecutive non-blank, non-special lines
  const buf = [t];
  i++;
  while (i < lines.length && lines[i].trim() && !/^(#|\||- |!\[|```|---$)/.test(lines[i].trim())) buf.push(lines[i++].trim());
  const text = buf.join(" ");
  if (text.startsWith("**Author:**")) {
    children.push(para(text, { alignment: AlignmentType.CENTER, run: { color: MUTED, size: 20 } }));
  } else if (/^\*\*Table \d/.test(text)) {
    children.push(para(text, { keepNext: true, spacing: { before: 120, after: 80 }, run: { size: 20 } }));
  } else {
    children.push(para(text, { alignment: AlignmentType.JUSTIFIED }));
  }
}

// ---------- document ----------
const heading = (size, before, after, color = INK) => ({
  run: { font: FONT, size, bold: true, color },
  paragraph: { spacing: { before, after }, keepNext: true, keepLines: true },
});

const doc = new Document({
  creator: "Anshu Kumar Sharma",
  title: "Forecasting the USD/INR Exchange Rate with Macroeconomic Indicators",
  description: "ML coursework report",
  styles: {
    default: { document: { run: { font: FONT, size: 22, color: INK } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", quickFormat: true,
        ...heading(34, 0, 160) },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        ...heading(28, 300, 120, "1F3864"), paragraph: { spacing: { before: 300, after: 120 }, keepNext: true, keepLines: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        ...heading(24, 200, 80, "2F5496"), paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, keepLines: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ font: FONT, size: 18, color: MUTED, children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES] })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length, "bytes"); });
