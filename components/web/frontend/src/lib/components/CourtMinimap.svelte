<script lang="ts">
	import type { AnnotationData, FrameAnnotation } from '$lib/types';
	import { onMount } from 'svelte';

	interface Props {
		annotations: AnnotationData | null;
		frame: number;
		visible: boolean;
		colorMode: 'team' | 'id';
		/** Accumulated court-position trails: player_id -> [[x,y], ...] */
		trails?: Record<number, [number, number][]>;
		/** Bindable: canvas element (for export) */
		minimapCanvas?: HTMLCanvasElement | null;
		/** Bindable: drag position from right edge */
		dragRight?: number;
		/** Bindable: drag position from bottom edge */
		dragBottom?: number;
	}

	let {
		annotations,
		frame,
		visible,
		colorMode,
		trails = {},
		minimapCanvas = $bindable<HTMLCanvasElement | null>(null),
		dragRight = $bindable(16),
		dragBottom = $bindable(16)
	}: Props = $props();

	let canvasEl: HTMLCanvasElement | undefined = $state(undefined);

	// Expose internal canvas element to parent via bindable prop
	$effect(() => {
		if (canvasEl) minimapCanvas = canvasEl;
	});

	let courtImage: HTMLImageElement | null = null;
	let imageLoaded = $state(false);

	onMount(() => {
		const img = new Image();
		img.src = '/nba_court.png';
		img.onload = () => {
			courtImage = img;
			imageLoaded = true;
		};
	});

	// NBA court dimensions in meters
	const COURT_W = 28.65;
	const COURT_H = 15.24;
	const CANVAS_W = 320;
	const CANVAS_H = 170;
	const PAD = 10;
	const DRAW_W = CANVAS_W - 2 * PAD;
	const DRAW_H = CANVAS_H - 2 * PAD;

	const TEAM_COLORS = ['#ff6b2b', '#3b82f6'];

	function courtToCanvas(xm: number, ym: number): [number, number] {
		const nx = (xm + COURT_W / 2) / COURT_W;
		const ny = 1 - (ym + COURT_H / 2) / COURT_H;
		return [PAD + nx * DRAW_W, PAD + ny * DRAW_H];
	}

	$effect(() => {
		const _vis = visible;
		const _frame = frame;
		const _ann = annotations;
		const _cm = colorMode;
		const _tr = trails;
		const _il = imageLoaded;
		if (!canvasEl) return;
		const ctx = canvasEl.getContext('2d');
		if (!ctx) return;
		drawBackground(ctx);
		drawPlayers(ctx);
	});

	function drawBackground(ctx: CanvasRenderingContext2D) {
		ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
		if (courtImage) {
			ctx.drawImage(courtImage, 0, 0, CANVAS_W, CANVAS_H);
		} else {
			ctx.fillStyle = '#1a3a2a';
			ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
		}
	}

	function playerColor(teamId: number | null, playerId: number): string {
		if (colorMode === 'id') return `hsl(${(Math.abs(playerId) * 137) % 360},70%,60%)`;
		return TEAM_COLORS[teamId ?? 0] ?? TEAM_COLORS[0];
	}

	function drawPlayers(ctx: CanvasRenderingContext2D) {
		if (!annotations) return;
		const fd: FrameAnnotation | undefined = annotations.frames[String(frame)];
		if (!fd) return;

		// Draw trails first (behind players)
		for (const p of fd.players) {
			if (p.player_id < 0) continue;
			const history = trails[p.player_id];
			if (!history || history.length < 2) continue;
			const color = playerColor(p.team_id, p.player_id);
			for (let i = 1; i < history.length; i++) {
				const alpha = (i / history.length) * 0.7;
				ctx.strokeStyle = color;
				ctx.globalAlpha = alpha;
				ctx.lineWidth = 2;
				ctx.beginPath();
				const [x1, y1] = courtToCanvas(history[i - 1][0], history[i - 1][1]);
				const [x2, y2] = courtToCanvas(history[i][0], history[i][1]);
				ctx.moveTo(x1, y1);
				ctx.lineTo(x2, y2);
				ctx.stroke();
			}
			ctx.globalAlpha = 1;
		}

		// Draw players on top
		for (const p of fd.players) {
			if (p.player_id < 0) continue;
			if (!p.court_position) continue;
			const [cx, cy] = courtToCanvas(p.court_position[0], p.court_position[1]);
			const color = playerColor(p.team_id, p.player_id);

			ctx.fillStyle = color;
			ctx.beginPath();
			ctx.arc(cx, cy, 5, 0, Math.PI * 2);
			ctx.fill();
			ctx.strokeStyle = 'rgba(255,255,255,0.6)';
			ctx.lineWidth = 1;
			ctx.stroke();

			ctx.fillStyle = '#fff';
			ctx.font = 'bold 8px Inter, sans-serif';
			ctx.textAlign = 'center';
			ctx.fillText(String(p.player_id), cx, cy - 7);
		}
	}

	// --- Drag logic ---
	let dragging = $state(false);
	let dragStartX = 0,
		dragStartY = 0;
	let dragInitRight = 0,
		dragInitBottom = 0;

	function onPointerDown(e: PointerEvent) {
		dragging = true;
		dragStartX = e.clientX;
		dragStartY = e.clientY;
		dragInitRight = dragRight;
		dragInitBottom = dragBottom;
		(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
		e.preventDefault();
	}

	function onPointerMove(e: PointerEvent) {
		if (!dragging) return;
		dragRight = Math.max(0, dragInitRight - (e.clientX - dragStartX));
		dragBottom = Math.max(0, dragInitBottom - (e.clientY - dragStartY));
	}

	function onPointerUp() {
		dragging = false;
	}
</script>

<div
	role="region"
	aria-label="Court minimap"
	class="absolute z-10 rounded-lg shadow-xl border border-[var(--color-border)] select-none"
	class:cursor-grabbing={dragging}
	class:cursor-grab={!dragging}
	style:display={visible ? '' : 'none'}
	style:right="{dragRight}px"
	style:bottom="{dragBottom}px"
	onpointerdown={onPointerDown}
	onpointermove={onPointerMove}
	onpointerup={onPointerUp}
>
	<canvas bind:this={canvasEl} width={CANVAS_W} height={CANVAS_H} class="block"></canvas>
</div>
