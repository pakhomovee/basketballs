<script lang="ts">
	import type { AnnotationData } from '$lib/types';

	interface Props {
		annotations: AnnotationData | null;
		currentTime: number;
		duration: number;
		onSeek?: (time: number) => void;
	}

	let { annotations, currentTime, duration, onSeek }: Props = $props();

	let track: HTMLDivElement;
	let dragging = $state(false);

	const tf = $derived(annotations?.metadata.total_frames ?? 1);
	const shots = $derived(annotations?.shot_events ?? []);
	const passes = $derived(annotations?.pass_events ?? []);
	const hasEvents = $derived(shots.length > 0 || passes.length > 0);

	function pctFrame(f: number): number {
		return (f / Math.max(tf, 1)) * 100;
	}

	function pctTime(t: number): number {
		if (!duration || duration <= 0) return 0;
		return (t / duration) * 100;
	}

	function passMid(p: { frame_start: number; frame_end: number }): number {
		return ((p.frame_start + p.frame_end) / 2 / Math.max(tf, 1)) * 100;
	}

	function teamHex(teamId: number): string {
		return teamId === 0 ? 'var(--color-team-orange)' : 'var(--color-team-blue)';
	}

	function seekFromPointer(clientX: number) {
		if (!track || !duration) return;
		const rect = track.getBoundingClientRect();
		const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
		onSeek?.(pct * duration);
	}

	function onPointerDown(e: PointerEvent) {
		dragging = true;
		(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
		seekFromPointer(e.clientX);
	}

	function onPointerMove(e: PointerEvent) {
		if (!dragging) return;
		seekFromPointer(e.clientX);
	}

	function onPointerUp() {
		dragging = false;
	}
</script>

<div
	bind:this={track}
	role="slider"
	tabindex="0"
	aria-label="Seek timeline"
	aria-valuemin={0}
	aria-valuemax={duration || 0}
	aria-valuenow={currentTime}
	class="relative w-full shrink-0 h-4 cursor-pointer select-none touch-none"
	onpointerdown={onPointerDown}
	onpointermove={onPointerMove}
	onpointerup={onPointerUp}
	onpointercancel={onPointerUp}
>
	<!-- Track background -->
	<div
		class="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-2.5 rounded-full bg-[var(--color-border)]/50 ring-1 ring-[var(--color-border)]/80"
	></div>

	<!-- Progress fill -->
	<div
		class="absolute left-0 top-1/2 -translate-y-1/2 h-2.5 rounded-full bg-[var(--color-accent)]/40"
		style:width="{pctTime(currentTime)}%"
	></div>

	{#if hasEvents}
		<!-- Shot segments -->
		<div
			class="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-2.5 rounded-full overflow-visible pointer-events-none"
		>
			{#each shots as s}
				{@const left = pctFrame(s.frame_start)}
				{@const rawW = ((s.frame_end - s.frame_start + 1) / Math.max(tf, 1)) * 100}
				{@const w = Math.max(rawW, 0.45)}
				<div
					class="absolute top-0 h-full rounded-sm opacity-95 overflow-hidden"
					style:left="{left}%"
					style:width="{w}%"
					style:background={s.is_make
						? 'linear-gradient(180deg, #34d399 0%, #059669 55%, #047857 100%)'
						: 'linear-gradient(180deg, #fb923c 0%, #ea580c 85%, #c2410c 100%)'}
					title="Shot · f{s.frame_start}–{s.frame_end}{s.is_make ? ' · make' : ''}"
				>
					{#if s.is_make && s.make_start != null && s.make_end != null}
						{@const segLo = s.frame_start}
						{@const segHi = s.frame_end}
						{@const mLo = Math.max(s.make_start, segLo)}
						{@const mHi = Math.min(s.make_end, segHi)}
						{@const innerLeft = ((mLo - segLo) / (segHi - segLo + 1)) * 100}
						{@const innerW = Math.max(((mHi - mLo + 1) / (segHi - segLo + 1)) * 100, 8)}
						<div
							class="absolute inset-y-0 rounded-sm bg-white/35 ring-1 ring-white/20"
							style:left="{innerLeft}%"
							style:width="{innerW}%"
						></div>
					{/if}
				</div>
			{/each}
		</div>

		<!-- Pass segments -->
		<div
			class="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-2.5 rounded-full overflow-visible pointer-events-none"
		>
			{#each passes as p}
				{@const left = pctFrame(p.frame_start)}
				{@const rawW = ((p.frame_end - p.frame_start + 1) / Math.max(tf, 1)) * 100}
				{@const w = Math.max(rawW, 0.35)}
				<div
					class="absolute top-0 h-full rounded-sm opacity-90"
					style:left="{left}%"
					style:width="{w}%"
					style:background={teamHex(p.team_id)}
					title="Pass · #{p.from_player_id}→#{p.to_player_id} · f{p.frame_start}–{p.frame_end}"
				></div>
			{/each}
		</div>
	{/if}

	<!-- Thumb -->
	<div
		class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 z-20 pointer-events-none
			w-3.5 h-3.5 rounded-full bg-[var(--color-accent)] border-2 border-white shadow-md
			transition-transform {dragging ? 'scale-125' : ''}"
		style:left="{pctTime(currentTime)}%"
	></div>

	<!-- Playhead line -->
	<div
		class="absolute top-0 bottom-0 w-px z-10 bg-white/70 pointer-events-none"
		style:left="{pctTime(currentTime)}%"
	></div>
</div>
