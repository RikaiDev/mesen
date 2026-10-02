/** Read-only public-page capture at four fixed browser widths. */
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const [selectionArg, yanaRootArg, outputArg] = process.argv.slice(2);
if (!selectionArg || !yanaRootArg || !outputArg) {
	throw new Error("usage: capture_public_holdout.mjs <selection.json> <yana-root> <output-dir>");
}
const selection = JSON.parse(readFileSync(selectionArg, "utf8"));
if (
	selection.schema !== "mesen.public-route-selection.v1" ||
	!Array.isArray(selection.routes) ||
	selection.routes.length < 1 ||
	selection.routes.length > 10
) {
	throw new Error("expected sealed public-page selection with one to ten routes");
}
const requireFromYana = createRequire(join(resolve(yanaRootArg), "e2e/package.json"));
const { chromium } = requireFromYana("@playwright/test");
const browser = await chromium.launch();
const widths = [375, 768, 1024, 1440];
try {
	for (const url of selection.routes) {
		const routeId = createHash("sha256").update(url).digest("hex").slice(0, 12);
		const routeDir = join(resolve(outputArg), `route-${routeId}`);
		mkdirSync(routeDir, { recursive: true });
		const viewportFacts = [];
		const consoleErrors = [];
		const httpFailed = [];
		for (const width of widths) {
			const page = await browser.newPage({ viewport: { width, height: 900 } });
			page.on("console", (message) => {
				if (message.type() === "error") consoleErrors.push(message.text().slice(0, 200));
			});
			page.on("pageerror", (error) => consoleErrors.push(String(error).slice(0, 200)));
			page.on("response", (response) => {
				if (response.status() >= 400) {
					httpFailed.push(`${response.status()} ${response.url().slice(0, 120)}`);
				}
			});
			const response = await page.goto(url, {
				waitUntil: "domcontentloaded",
				timeout: 30000,
			});
			if (!response || response.status() >= 400) {
				throw new Error(`Public route ${routeId} returned ${response?.status() ?? "no response"}`);
			}
			await page.waitForTimeout(1500);
			await page
				.waitForFunction(() => [...document.images].every((image) => image.complete), {
					timeout: 10000,
				})
				.catch(() => null);
			await page.screenshot({ path: join(routeDir, `${width}.png`), fullPage: false });
			const facts = await page.evaluate(() => ({
				title: document.title,
				docScrollWidth: document.documentElement.scrollWidth,
				brokenImages: [...document.images]
					.filter((image) => image.complete && image.naturalWidth === 0)
					.map((image) => image.currentSrc || image.src)
					.slice(0, 20),
			}));
			viewportFacts.push({ width, screenshot: `${width}.png`, ...facts });
			await page.close();
		}
		writeFileSync(
			join(routeDir, "state.json"),
			JSON.stringify(
				{
					witnessVersion: 2,
					product: selection.site_family,
					route: new URL(url).pathname,
					url,
					context: { cohort: "general_public", modality: "desktop_web", interaction_mode: "pointer" },
					viewportFacts,
					consoleErrors: [...new Set(consoleErrors)],
					httpFailed: [...new Set(httpFailed)],
				},
				null,
				2,
			),
		);
		console.log(JSON.stringify({ routeId, widths: viewportFacts.length, httpErrors: httpFailed.length }));
	}
} finally {
	await browser.close();
}
