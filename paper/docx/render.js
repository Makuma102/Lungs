// Render paper/docx/blocks.json (produced by scripts/build_journal.py) into a .docx.
// Block types: title, authors, h1, h2, h3, p, bullets, table, img, caption, refs, note, pagebreak.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, ShadingType, LevelFormat, PageBreak, Footer, PageNumber,
} = require("docx");

const [, , inJson, outDocx] = process.argv;
const blocks = JSON.parse(fs.readFileSync(inJson, "utf8"));
const FONT = "Times New Roman";
const PAGE_W = 11906, MARGIN = 1134;              // A4, 2 cm margins
const TEXT_W = PAGE_W - 2 * MARGIN;               // 9638 DXA

function runs(r) {
  // r: string or array of {text,bold,italic,sup,sub}
  if (typeof r === "string") r = [{ text: r }];
  return r.map((x) => new TextRun({ text: x.text, bold: !!x.bold, italics: !!x.italic,
    superScript: !!x.sup, subScript: !!x.sub, font: FONT, size: x.size || 22 }));
}
const para = (r, o = {}) => new Paragraph({ children: runs(r), alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after ?? 120, line: 276 }, ...o.extra });

const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

function table(b) {
  const n = b.header.length;
  const widths = b.widths ? b.widths.map((w) => Math.round(w * TEXT_W)) : Array(n).fill(Math.floor(TEXT_W / n));
  widths[widths.length - 1] += TEXT_W - widths.reduce((a, c) => a + c, 0);
  const cell = (text, i, isHead, last) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    borders: { top: isHead ? thin : none, bottom: (isHead || last) ? thin : none, left: none, right: none },
    shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: "F2F2F2" } : undefined,
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(typeof text === "string" ? [{ text, bold: isHead, size: 18 }] : text.map((t) => ({ ...t, size: 18 }))) })],
  });
  const rows = [new TableRow({ tableHeader: true, children: b.header.map((h, i) => cell(h, i, true, false)) })];
  b.rows.forEach((r, ri) => rows.push(new TableRow({ children: r.map((c, i) => cell(c, i, false, ri === b.rows.length - 1)) })));
  return new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, rows });
}

function image(b) {
  const data = fs.readFileSync(b.path);
  // read PNG size from the IHDR chunk to keep the aspect ratio
  const w = data.readUInt32BE(16), h = data.readUInt32BE(20);
  const maxW = (b.width || 1.0) * 6.3 * 96;       // fraction of text width (~6.3 in)
  const sc = maxW / w;
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data, transformation: { width: Math.round(w * sc), height: Math.round(h * sc) } })] });
}

const children = [];
for (const b of blocks) {
  switch (b.t) {
    case "title": children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new TextRun({ text: b.text, bold: true, font: FONT, size: 32 })] })); break;
    case "authors": children.push(para(b.text, { align: AlignmentType.CENTER, after: 240 })); break;
    case "note": children.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 160 },
      shading: { type: ShadingType.CLEAR, color: "auto", fill: "FFF4D6" },
      children: [new TextRun({ text: b.text, italics: true, font: FONT, size: 20 })] })); break;
    case "h1": children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 120 },
      children: [new TextRun({ text: b.text, bold: true, font: FONT, size: 26 })] })); break;
    case "h2": children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 180, after: 80 },
      children: [new TextRun({ text: b.text, bold: true, font: FONT, size: 23 })] })); break;
    case "h3": children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 120, after: 60 },
      children: [new TextRun({ text: b.text, italics: true, font: FONT, size: 22 })] })); break;
    case "p": children.push(para(b.runs || b.text)); break;
    case "bullets": b.items.forEach((it) => children.push(new Paragraph({ numbering: { reference: "bul", level: 0 },
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 60 }, children: runs(it) }))); break;
    case "caption": children.push(para(b.runs || b.text, { align: AlignmentType.LEFT, after: 160 })); break;
    case "table":
      if (b.caption) children.push(new Paragraph({ spacing: { before: 160, after: 80 }, children: runs(b.caption.map ? b.caption : [{ text: b.caption }]).map((r) => r) }));
      children.push(table(b));
      children.push(new Paragraph({ spacing: { after: 120 }, children: [] })); break;
    case "img": children.push(image(b));
      if (b.caption) children.push(new Paragraph({ spacing: { after: 200 }, alignment: AlignmentType.JUSTIFIED,
        children: runs(b.caption).map((r) => r) })); break;
    case "refs": b.items.forEach((it, i) => children.push(new Paragraph({ spacing: { after: 60 }, indent: { left: 440, hanging: 440 },
      children: [new TextRun({ text: `[${i + 1}] `, font: FONT, size: 20 }), new TextRun({ text: it, font: FONT, size: 20 })] }))); break;
    case "pagebreak": children.push(new Paragraph({ children: [new PageBreak()] })); break;
    default: throw new Error("unknown block " + b.t);
  }
}

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outDocx, buf); console.log("wrote", outDocx); });
