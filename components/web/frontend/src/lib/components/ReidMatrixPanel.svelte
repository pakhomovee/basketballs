<script lang="ts">
	import type { ReidMatrix } from '$lib/types';

	interface Props {
		matrix: ReidMatrix | null;
		visible: boolean;
	}

	let { matrix, visible }: Props = $props();

	let canvas = $state<HTMLCanvasElement | undefined>();
	let tooltip = $state({ show: false, x: 0, y: 0, text: '' });

	const CELL = 40;
	const HEADER = 32;
	const PAD = 8;

	$effect(() => {
		// Read both reactive deps at the top so the effect re-runs whenever
		// either changes (including visible toggling back on while matrix is set).
		const m = matrix;
		const vis = visible;
		if (!canvas || !m || !vis) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		drawMatrix(ctx, m);
	});

	function distToColor(d: number): string {
		// 0.0 = green, 0.5 = yellow, 1.0+ = red
		const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
		const t = Math.min(d, 1.0);
		if (t <= 0.5) {
			const f = t / 0.5;
			return `rgb(${clamp(34 + (250 - 34) * f)},${clamp(197 + (204 - 197) * f)},${clamp(94 - 94 * f)})`;
		} else {
			const f = (t - 0.5) / 0.5;
			return `rgb(${clamp(250 + (239 - 250) * f)},${clamp(204 * (1 - f))},${clamp(68 * f)})`;
		}
	}

	function drawMatrix(ctx: CanvasRenderingContext2D, m: ReidMatrix) {
		const rows = m.row_ids.length;
		const cols = m.col_ids.length;
		const w = PAD + HEADER + cols * CELL + PAD;
		const h = PAD + HEADER + rows * CELL + PAD;

		if (!canvas) return;
		canvas.width = w;
		canvas.height = h;

		ctx.fillStyle = '#1a2030';
		ctx.fillRect(0, 0, w, h);

		ctx.font = 'bold 11px Inter, sans-serif';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';

		// Column headers (current frame player IDs)
		ctx.fillStyle = '#8899aa';
		for (let c = 0; c < cols; c++) {
			const cx = PAD + HEADER + c * CELL + CELL / 2;
			ctx.fillText(String(m.col_ids[c]), cx, PAD + HEADER / 2);
		}

		// Row headers (prev frame player IDs)
		for (let r = 0; r < rows; r++) {
			const cy = PAD + HEADER + r * CELL + CELL / 2;
			ctx.fillText(String(m.row_ids[r]), PAD + HEADER / 2, cy);
		}

		// Cells
		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < cols; c++) {
				const d = m.distances[r][c];
				const x = PAD + HEADER + c * CELL;
				const y = PAD + HEADER + r * CELL;

				ctx.fillStyle = distToColor(d);
				ctx.beginPath();
				ctx.roundRect(x + 1, y + 1, CELL - 2, CELL - 2, 4);
				ctx.fill();

				// Diagonal highlight border
				if (m.row_ids[r] === m.col_ids[c]) {
					ctx.strokeStyle = 'rgba(255,255,255,0.5)';
					ctx.lineWidth = 1.5;
					ctx.beginPath();
					ctx.roundRect(x + 1, y + 1, CELL - 2, CELL - 2, 4);
					ctx.stroke();
				}

				// Value text
				ctx.fillStyle = d > 0.6 ? '#fff' : '#000';
				ctx.font = '10px Inter, sans-serif';
				ctx.fillText(d.toFixed(2), x + CELL / 2, y + CELL / 2);
			}
		}

		// Axis labels
		ctx.save();
		ctx.fillStyle = '#556677';
		ctx.font = '9px Inter, sans-serif';
		ctx.textAlign = 'center';
		ctx.fillText('Current frame →', PAD + HEADER + (cols * CELL) / 2, PAD + HEADER / 2 - 10);
		ctx.translate(PAD + HEADER / 2 - 10, PAD + HEADER + (rows * CELL) / 2);
		ctx.rotate(-Math.PI / 2);
		ctx.fillText('Previous frame →', 0, 0);
		ctx.restore();
	}

	function onMouseMove(e: MouseEvent) {
		if (!matrix || !canvas) {
			tooltip.show = false;
			return;
		}
		const rect = canvas.getBoundingClientRect();
		const mx = e.clientX - rect.left;
		const my = e.clientY - rect.top;

		const col = Math.floor((mx - PAD - HEADER) / CELL);
		const row = Math.floor((my - PAD - HEADER) / CELL);

		if (row >= 0 && row < matrix.row_ids.length && col >= 0 && col < matrix.col_ids.length) {
			const d = matrix.distances[row][col];
			tooltip = {
				show: true,
				x: e.clientX,
				y: e.clientY,
				text: `ID ${matrix.row_ids[row]} → ${matrix.col_ids[col]}: ${d.toFixed(4)}`
			};
		} else {
			tooltip.show = false;
		}
	}
</script>

<div style:display={visible ? '' : 'none'}>
	{#if matrix}
		<div class="mt-4">
			<h3 class="text-sm font-semibold text-[var(--color-text-muted)] mb-2">
				ReID Distance Matrix
				<span class="font-normal text-xs">(prev frame → current frame)</span>
			</h3>
			<div
				class="relative inline-block bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)] p-1"
			>
				<canvas
					bind:this={canvas}
					onmousemove={onMouseMove}
					onmouseleave={() => {
						tooltip.show = false;
					}}
					class="cursor-crosshair"
				></canvas>
			</div>
		</div>
	{:else}
		<p class="mt-4 text-xs text-[var(--color-text-muted)] italic">
			No ReID data at this frame (seek forward)
		</p>
	{/if}
</div>

{#if tooltip.show}
	<div
		class="fixed z-50 px-2 py-1 text-xs rounded bg-black/80 text-white pointer-events-none"
		style="left: {tooltip.x + 12}px; top: {tooltip.y - 8}px;"
	>
		{tooltip.text}
	</div>
{/if}
