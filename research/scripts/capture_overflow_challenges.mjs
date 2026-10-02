/** Capture clean/mutated browser pairs for sealed cross-site overflow cases. */
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const [selectionArg, yanaRootArg, outputArg, siteFilter] = process.argv.slice(2);
if (!selectionArg || !yanaRootArg || !outputArg) {
	throw new Error("usage: capture_overflow_challenges.mjs <selection.json> <yana-root> <output-dir>");
}
const selection = JSON.parse(readFileSync(selectionArg, "utf8"));
if (selection.schema !== "mesen.overflow-challenge-selection.v1" || selection.cases.length !== 40) {
	throw new Error("expected frozen forty-case overflow selection");
}
const requireFromYana = createRequire(join(resolve(yanaRootArg), "e2e/package.json"));
const { chromium } = requireFromYana("@playwright/test");
const browser = await chromium.launch();
try {
	const cases = siteFilter
		? selection.cases.filter((challenge) => challenge.site_family === siteFilter)
		: selection.cases;
	if (cases.length === 0) throw new Error("site filter selected no challenges");
	for (const challenge of cases) {
		const outputDir = join(resolve(outputArg), challenge.id);
		mkdirSync(outputDir, { recursive: true });
		const page = await browser.newPage({
			viewport: { width: challenge.viewport_width, height: 900 },
		});
		const consoleErrors = [];
		const httpErrors = [];
		page.on("console", (message) => {
			if (message.type() === "error") consoleErrors.push(message.text().slice(0, 200));
		});
		page.on("pageerror", (error) => consoleErrors.push(String(error).slice(0, 200)));
		page.on("response", (response) => {
			if (response.status() >= 400) {
				httpErrors.push(`${response.status()} ${response.url().slice(0, 120)}`);
			}
		});
		const response = await page.goto(challenge.url, {
			waitUntil: "domcontentloaded",
			timeout: 30000,
		});
		if (!response || response.status() >= 400) {
			throw new Error(`Challenge ${challenge.id} returned ${response?.status() ?? "no response"}`);
		}
		if (new URL(page.url()).pathname !== new URL(challenge.url).pathname) {
			throw new Error(`Challenge ${challenge.id} redirected to a different route`);
		}
		await page.waitForTimeout(1200);
		await page
			.waitForFunction(() => [...document.images].every((image) => image.complete), {
				timeout: 10000,
			})
			.catch(() => null);
		const before = await page.evaluate(() => document.documentElement.scrollWidth);
		await page.screenshot({ path: join(outputDir, "clean.png"), fullPage: false });
		if (before > challenge.viewport_width) {
			throw new Error(`Challenge ${challenge.id} was already document-overflowing`);
		}
		const css = `html, body { min-width: ${challenge.injected_min_width}px !important; }`;
		let mutationMethod = "local browser style tag";
		let styleAsset = null;
		if (challenge.site_family === "govuk-help") {
			mutationMethod = "same-origin stylesheet response append in local browser";
			await page.route("**/assets/frontend/application-*.css", async (route) => {
				const asset = await route.fetch();
				const original = await asset.body();
				const modified = Buffer.concat([original, Buffer.from(`\n${css}\n`)]);
				styleAsset = {
					url: route.request().url(),
					original_sha256: createHash("sha256").update(original).digest("hex"),
					modified_sha256: createHash("sha256").update(modified).digest("hex"),
				};
				await route.fulfill({ response: asset, body: modified });
			});
			await page.reload({ waitUntil: "domcontentloaded", timeout: 30000 });
			if (!styleAsset) {
				throw new Error(`Challenge ${challenge.id} had no same-origin application stylesheet`);
			}
		} else {
			await page.addStyleTag({ content: css });
		}
		await page.waitForTimeout(150);
		const after = await page.evaluate(() => document.documentElement.scrollWidth);
		await page.screenshot({ path: join(outputDir, "mutated.png"), fullPage: false });
		if (after <= challenge.viewport_width) {
			throw new Error(`Challenge ${challenge.id} did not produce document overflow`);
		}
		writeFileSync(
			join(outputDir, "state.json"),
			JSON.stringify(
				{
					schema: "mesen.paired-overflow-witness.v1",
					id: challenge.id,
					site_family: challenge.site_family,
					route_family: challenge.route_family,
					requested_url: challenge.url,
					final_url: page.url(),
					viewport_width: challenge.viewport_width,
					baseline_document_scroll_width: before,
					mutated_document_scroll_width: after,
					injected_css: css,
					mutation_method: mutationMethod,
					style_asset: styleAsset,
					clean_screenshot: "clean.png",
					mutated_screenshot: "mutated.png",
					console_errors: [...new Set(consoleErrors)],
					http_errors: [...new Set(httpErrors)],
				},
				null,
				2,
			),
		);
		console.log(JSON.stringify({ id: challenge.id, before, after }));
		await page.close();
	}
} finally {
	await browser.close();
}
