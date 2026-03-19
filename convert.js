/**
 * convert.js
 * Converts a structured lab-report markdown file into a
 * LaTeX-inspired academic PDF using marked + puppeteer.
 *
 * Usage:
 *   node convert.js <input.md> [output.pdf]
 *
 * The first ## heading must follow the pattern:
 *   ## <Course & Lab Title>  <StudentID>  <Student Name>
 *   e.g. ## Cloud Computing Lab 7 2023BCY0048 Akshat Ajit Saraswat
 */

"use strict";

const fs   = require("fs");
const path = require("path");
const os   = require("os");
const { marked, Renderer } = require("marked");
const puppeteer = require("puppeteer");

/* ──────────────────────────────────────────────
   1. METADATA EXTRACTION
   ────────────────────────────────────────────── */

/**
 * Parses the special first heading line:
 * "Cloud Computing Lab 7 2023BCY0048 Akshat Ajit Saraswat"
 * Returns { courseAndLab, studentId, studentName }
 */
function extractMetadata(rawFirstLine) {
  // Strip leading # characters and trim
  const text = rawFirstLine.replace(/^#+\s*/, "").trim();

  // Student IDs typically look like 2023BCY0048 – a 4-digit year
  // followed by uppercase letters then digits
  const idRx = /\b(\d{4}[A-Z]{2,5}\d{3,6})\b/;
  const match = text.match(idRx);

  if (match) {
    const idStart = text.indexOf(match[1]);
    const courseAndLab = text.slice(0, idStart).trim();
    const studentId    = match[1];
    const studentName  = text.slice(idStart + match[1].length).trim();
    return { courseAndLab, studentId, studentName };
  }

  // Fallback – no recognisable ID pattern
  return { courseAndLab: text, studentId: "—", studentName: "—" };
}

/* ──────────────────────────────────────────────
   2. CUSTOM MARKED RENDERER
   ────────────────────────────────────────────── */

let figureCounter = 0;

function buildRenderer(imageBasePath) {
  const renderer = new Renderer();

  /* Images → <figure> with caption */
  renderer.image = function ({ href, title, text }) {
    figureCounter += 1;

    // Resolve image path relative to the markdown file and convert to a
    // file:// URL so Puppeteer's setContent() can actually load local files.
    const resolvedSrc = imageBasePath
      ? `file://${path.resolve(imageBasePath, href)}`
      : href;
    console.log(`Processing image: ${resolvedSrc}`);

    // Use title attribute first, then alt text
    const captionText = title || text || "";
    const num = figureCounter;

    return `
      <figure class="lab-figure">
        <img src="${resolvedSrc}" alt="${text}" />
        <figcaption><strong>Figure ${num}:</strong> ${captionText}</figcaption>
      </figure>`;
  };

  /* hr → styled section divider */
  renderer.hr = function () {
    return `<div class="section-divider"></div>`;
  };

  /* h2 → numbered section heading */
  renderer.heading = function ({ text, depth }) {
    const tag = `h${depth}`;
    const cls = `heading-${depth}`;
    return `<${tag} class="${cls}">${text}</${tag}>`;
  };

  return renderer;
}

/* ──────────────────────────────────────────────
   3. HTML TEMPLATE
   ────────────────────────────────────────────── */

function buildHtml(metadata, bodyHtml, css) {
  const today = new Date().toLocaleDateString("en-GB", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  // Derive a clean course name and lab number for the title page
  // e.g. "Cloud Computing Lab 7" → course: "Cloud Computing", lab: "Lab 7"
  const labMatch  = metadata.courseAndLab.match(/(Lab\s*\d+)/i);
  const labNumber = labMatch ? labMatch[1] : "";
  const courseName = "CBS321 Cloud Computing and Security"

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
${css}
  </style>
</head>
<body>

  <!-- ═══════════════ TITLE PAGE ═══════════════ -->
  <div class="title-page">
    <div class="title-page__inner">
      <div class="title-rule top-rule"></div>

      <p class="title-page__course">${escHtml(courseName)}</p>
      <h1 class="title-page__labtitle">${escHtml(labNumber)}</h1>
      <p class="title-page__subtitle">Lab Report</p>

      <div class="title-rule bottom-rule"></div>

      <table class="title-meta">
        <tbody>
          <tr>
            <td class="meta-label">Name</td>
            <td class="meta-colon">:</td>
            <td class="meta-value">${escHtml(metadata.studentName)}</td>
          </tr>
          <tr>
            <td class="meta-label">Roll&nbsp;Number</td>
            <td class="meta-colon">:</td>
            <td class="meta-value">${escHtml(metadata.studentId)}</td>
          </tr>
          <tr>
            <td class="meta-label">Date</td>
            <td class="meta-colon">:</td>
            <td class="meta-value">${today}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ═══════════════ REPORT BODY ═══════════════ -->
  <div class="report-body">
    ${bodyHtml}
  </div>

</body>
</html>`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ──────────────────────────────────────────────
   4. PDF GENERATION
   ────────────────────────────────────────────── */

const footerTpl = `
  <div style="
    font-family: 'Times New Roman', Times, serif;
    font-size: 9pt;
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 25mm;
    box-sizing: border-box;
    color: #000;
    border-top: 0.5pt solid #000;
    padding-top: 4px;
  ">
    <span style="font-style: italic; font-size: 8pt;">2023BCY0048</span>
    <span class="pageNumber"></span>
  </div>`;

const headerTpl = `<div></div>`;

async function generatePdf(html, outputPath) {
  const browser = await puppeteer.launch({
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  // Write HTML to a temp file and navigate via file:// so Chromium's
  // file-origin security model allows loading sibling file:// images.
  const tmpHtml = path.join(os.tmpdir(), `labai_report_${Date.now()}.html`);

  try {
    fs.writeFileSync(tmpHtml, html, "utf-8");
    const page = await browser.newPage();

    await page.goto(`file://${tmpHtml}`, { waitUntil: "networkidle0" });

    // Wait for images to load
    await page.evaluate(() =>
      Promise.all(
        Array.from(document.images).map((img) =>
          img.complete
            ? Promise.resolve()
            : new Promise((res) => { img.onload = res; img.onerror = res; })
        )
      )
    );

    await page.pdf({
      path: outputPath,
      format: "A4",
      printBackground: true,
      margin: {
        top: "25mm",
        bottom: "22mm",
        left: "32mm",   // wider left margin — mimics LaTeX default
        right: "25mm",
      },
      displayHeaderFooter: true,
      headerTemplate: headerTpl,
      footerTemplate: footerTpl,
    });
  } finally {
    await browser.close();
    try { fs.unlinkSync(tmpHtml); } catch (_) {}
  }
}

/* ──────────────────────────────────────────────
   5. MAIN
   ────────────────────────────────────────────── */

async function main() {
  const [inputFile, outputFile = "report.pdf"] = process.argv.slice(2);

  if (!inputFile) {
    console.error("Usage: node convert.js <input.md> [output.pdf]");
    process.exit(1);
  }

  if (!fs.existsSync(inputFile)) {
    console.error(`File not found: ${inputFile}`);
    process.exit(1);
  }

  const rawContent = fs.readFileSync(inputFile, "utf-8");
  const lines = rawContent.split("\n");

  // Separate the special first heading from the body
  let metadataLine = "";
  let bodyLines    = [];
  let metadataTaken = false;

  for (const line of lines) {
    if (!metadataTaken && /^##\s/.test(line)) {
      metadataLine  = line;
      metadataTaken = true;
    } else {
      bodyLines.push(line);
    }
  }

  const metadata    = extractMetadata(metadataLine);
  const bodyContent = bodyLines.join("\n").trim();

  // Image paths are relative to the markdown file's directory
  const imageBasePath = path.dirname(path.resolve(inputFile));

  figureCounter = 0; // reset before each run
  marked.use({ renderer: buildRenderer(imageBasePath) });

  const bodyHtml = marked.parse(bodyContent);

  // Load CSS from same directory as this script
  const cssPath = path.join(__dirname, "report.css");
  if (!fs.existsSync(cssPath)) {
    console.error(`report.css not found at: ${cssPath}`);
    process.exit(1);
  }
  const css = fs.readFileSync(cssPath, "utf-8");

  const html = buildHtml(metadata, bodyHtml, css);

  console.log(`Generating PDF → ${outputFile} …`);
  await generatePdf(html, outputFile);
  console.log("Done.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
