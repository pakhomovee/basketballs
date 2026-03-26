/** Shared canvas drawing utilities for annotation overlays. */

import type { FrameAnnotation, PlayerAnnotation, ToggleState } from './types';

const TEAM_COLORS = ['#ff6b2b', '#3b82f6'];
const TEAM_COLORS_ALPHA = ['rgba(255,107,43,0.3)', 'rgba(59,130,246,0.3)'];

// COCO-17 limb pairs for pose skeleton
const LIMBS: [number, number][] = [
	[5, 6],
	[5, 7],
	[7, 9],
	[6, 8],
	[8, 10],
	[11, 12],
	[11, 13],
	[13, 15],
	[12, 14],
	[14, 16]
];

export function playerColor(p: PlayerAnnotation, colorMode: 'team' | 'id', alpha = false): string {
	if (p.player_id < 0) {
		return alpha ? 'rgba(140,140,140,0.3)' : '#8c8c8c';
	}
	if (colorMode === 'id') {
		const hue = (Math.abs(p.player_id) * 137) % 360;
		return alpha ? `hsla(${hue},70%,60%,0.3)` : `hsl(${hue},70%,60%)`;
	}
	const idx = p.team_id ?? 0;
	return alpha
		? (TEAM_COLORS_ALPHA[idx] ?? TEAM_COLORS_ALPHA[0])
		: (TEAM_COLORS[idx] ?? TEAM_COLORS[0]);
}

function drawSkeleton(
	ctx: CanvasRenderingContext2D,
	p: PlayerAnnotation,
	sx: number,
	sy: number
): void {
	if (!p.skeleton) return;
	const confThresh = 0.3;

	ctx.strokeStyle = '#ef4444';
	ctx.lineWidth = 2;
	for (const [i, j] of LIMBS) {
		if (i >= p.skeleton.length || j >= p.skeleton.length) continue;
		const [x1, y1, c1] = p.skeleton[i];
		const [x2, y2, c2] = p.skeleton[j];
		if (c1 < confThresh || c2 < confThresh) continue;
		ctx.beginPath();
		ctx.moveTo(x1 * sx, y1 * sy);
		ctx.lineTo(x2 * sx, y2 * sy);
		ctx.stroke();
	}

	ctx.fillStyle = '#22c55e';
	for (const [x, y, c] of p.skeleton) {
		if (c < confThresh) continue;
		ctx.beginPath();
		ctx.arc(x * sx, y * sy, 3, 0, Math.PI * 2);
		ctx.fill();
	}
}

/**
 * Draw all enabled annotation layers for one frame onto ctx.
 * sx/sy are scale factors from annotation coordinate space to canvas pixels.
 */
export function drawAnnotations(
	ctx: CanvasRenderingContext2D,
	fd: FrameAnnotation,
	toggles: ToggleState,
	sx: number,
	sy: number
): void {
	// 1. Segmentation masks
	if (toggles.masks) {
		for (const p of fd.players) {
			if (!p.mask_polygon) continue;
			if (p.player_id < 0 && toggles.colorMode !== 'id') continue;
			ctx.save();
			// Clip to the player's bounding box so the mask never bleeds outside it.
			if (p.bbox) {
				const [bx1, by1, bx2, by2] = p.bbox;
				ctx.beginPath();
				ctx.rect(bx1 * sx, by1 * sy, (bx2 - bx1) * sx, (by2 - by1) * sy);
				ctx.clip();
			}
			ctx.fillStyle = playerColor(p, toggles.colorMode, true);
			ctx.beginPath();
			for (let i = 0; i < p.mask_polygon.length; i++) {
				const [x, y] = p.mask_polygon[i];
				if (i === 0) ctx.moveTo(x * sx, y * sy);
				else ctx.lineTo(x * sx, y * sy);
			}
			ctx.closePath();
			ctx.fill();
			ctx.restore();
		}
	}

	// 2. Player bounding boxes
	if (toggles.bboxes) {
		for (const p of fd.players) {
			if (!p.bbox) continue;
			if (p.player_id < 0 && toggles.colorMode !== 'id') continue;
			const [x1, y1, x2, y2] = p.bbox;
			ctx.strokeStyle = playerColor(p, toggles.colorMode);
			ctx.lineWidth = 2;
			ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
		}
	}

	// 3. Pose skeletons
	if (toggles.skeletons) {
		for (const p of fd.players) {
			drawSkeleton(ctx, p, sx, sy);
		}
	}

	// 4. Ball — filled orange circle with white outline
	if (toggles.ball) {
		for (const b of fd.balls) {
			if (!b.bbox) continue;
			const [x1, y1, x2, y2] = b.bbox;
			const cx = ((x1 + x2) / 2) * sx;
			const cy = ((y1 + y2) / 2) * sy;
			const r = Math.max(6, ((x2 - x1) / 2) * sx);
			ctx.fillStyle = '#ff8c00';
			ctx.beginPath();
			ctx.arc(cx, cy, r, 0, Math.PI * 2);
			ctx.fill();
			ctx.strokeStyle = 'rgba(255,255,255,0.8)';
			ctx.lineWidth = 1.5;
			ctx.stroke();
		}
	}

	// 5. Jersey numbers
	if (toggles.jerseyNumbers) {
		for (const p of fd.players) {
			if (!p.jersey_number || !p.bbox) continue;
			const [x1, y1, x2] = p.bbox;
			const cx = ((x1 + x2) / 2) * sx;
			const ty = y1 * sy - 6;
			ctx.font = 'bold 13px Inter, sans-serif';
			ctx.textAlign = 'center';
			ctx.fillStyle = '#fff';
			ctx.strokeStyle = 'rgba(0,0,0,0.7)';
			ctx.lineWidth = 3;
			ctx.strokeText(`#${p.jersey_number}`, cx, ty);
			ctx.fillText(`#${p.jersey_number}`, cx, ty);
		}
	}

	// 6. Player ID labels with background pill
	if (toggles.playerIds) {
		for (const p of fd.players) {
			if (!p.bbox || p.player_id < 0) continue;
			const [x1, y1] = p.bbox;
			const px = x1 * sx + 4;
			const py = y1 * sy + 16;
			ctx.font = 'bold 14px Inter, sans-serif';
			ctx.textAlign = 'left';
			const text = String(p.player_id);
			const tw = ctx.measureText(text).width;
			ctx.fillStyle = playerColor(p, toggles.colorMode);
			ctx.beginPath();
			ctx.roundRect(px - 3, py - 13, tw + 6, 17, 3);
			ctx.fill();
			ctx.fillStyle = '#fff';
			ctx.fillText(text, px, py);
		}
	}
}
