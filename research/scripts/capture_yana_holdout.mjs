/** Capture the preselected public Yana routes with the existing browser witness. */
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";

const [selectionArg, yanaRootArg, outputArg] = process.argv.slice(2);
if (!selectionArg || !yanaRootArg || !outputArg) {
	throw new Error("usage: capture_yana_holdout.mjs <selection.json> <yana-root> <output-dir>");
}
const selection = JSON.parse(readFileSync(selectionArg, "utf8"));
if (selection.schema !== "mesen.route-selection.v1" || selection.routes.length !== 10) {
	throw new Error("expected the sealed ten-route Yana selection");
}
const yanaRoot = resolve(yanaRootArg);
const script = join(yanaRoot, "e2e/visual/mesen-witness.mjs");
for (const route of selection.routes) {
	const routeId = createHash("sha256").update(route).digest("hex").slice(0, 12);
	const routeOutput = join(resolve(outputArg), `route-${routeId}`);
	const code = await new Promise((done, fail) => {
		const child = spawn(
			process.execPath,
			[script, "http://45.32.56.160", routeOutput, route],
			{ cwd: yanaRoot, stdio: "inherit" },
		);
		child.once("error", fail);
		child.once("exit", (exitCode) => done(exitCode ?? 1));
	});
	if (code !== 0) {
		throw new Error(`witness capture failed for route ${routeId} with code ${code}`);
	}
}
