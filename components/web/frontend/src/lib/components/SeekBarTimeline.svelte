<script lang="ts">
	import type { AnnotationData } from '$lib/types';

	interface Props {
		annotations: AnnotationData | null;
		currentTime: number;
		duration: number;
	}

	let { annotations, currentTime, duration }: Props = $props();

	const tf = $derived(annotations?.metadata.total_frames ?? 1);
	const shots = $derived(annotations?.shot_events ?? []);
	const passes = $derived(annotations?.pass_events ?? []);

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
</script>

{#if annotations && (shots.length > 0 || passes.length > 0)}
	<div
		class="relative w-full h-8 shrink-0 rounded-lg overflow-visible pb-5 mb-0.5"
		aria-label="Shot and pass timeline"
	>
		<!-- Playhead -->
		<div
			class="absolute top-0 bottom-0 w-px z-10 bg-white/90 shadow-[0_0_6px_rgba(255,255,255,0.5)] pointer-events-none"
			style:left="{pctTime(currentTime)}%"
		></div>

		<!-- Track background -->
		<div
			class="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-2.5 rounded-full bg-[var(--color-border)]/50 ring-1 ring-[var(--color-border)]/80"
		></div>

		<!-- Shot segments (middle layer); make window drawn as brighter inner band -->
		<div
			class="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-2.5 rounded-full overflow-visible pointer-events-none"
		>
			{#each shots as s}
				{@const left = pctFrame(s.frame_start)}
				{@const rawW = ((s.frame_end - s.frame_start + 1) / Math.max(tf, 1)) * 100}
				{@const w = Math.max(rawW, 0.45)}
				<div
					class="absolute top-0 h-full rounded-sm transition-opacity hover:opacity-100 opacity-95 overflow-hidden"
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

		<!-- Pass markers (top ticks) -->
		<div class="absolute left-0 right-0 top-0 h-3 pointer-events-none">
			{#each passes as p}
				<div
					class="absolute bottom-0 w-px h-2.5 rounded-t-sm opacity-95 -translate-x-1/2 shadow-[0_0_4px_currentColor]"
					style:left="{passMid(p)}%"
					style:background={teamHex(p.team_id)}
					title="Pass · #{p.from_player_id}→#{p.to_player_id}"
				></div>
			{/each}
		</div>

		<!-- Legend -->
		<div
			class="absolute -bottom-5 left-0 right-0 flex justify-center gap-3 text-[9px] uppercase tracking-wider text-[var(--color-text-muted)]"
		>
			<span class="inline-flex items-center gap-1"
				><i class="inline-block w-2 h-2 rounded-sm bg-gradient-to-b from-amber-400 to-orange-600"
				></i> Shot</span
			>
			<span class="inline-flex items-center gap-1"
				><i class="inline-block w-2 h-2 rounded-sm bg-gradient-to-b from-emerald-400 to-emerald-800"
				></i> Make</span
			>
			<span class="inline-flex items-center gap-1"
				><i class="inline-block w-0.5 h-2 bg-[var(--color-team-orange)]"></i> Pass</span
			>
		</div>
	</div>
{/if}
