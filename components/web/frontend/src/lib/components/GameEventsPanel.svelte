<script lang="ts">
	import type { PassEventAnnotation, ShotEventAnnotation } from '$lib/types';

	const TEAM_COLORS = ['#ff6b2b', '#3b82f6'];

	type Row =
		| { kind: 'pass'; pass: PassEventAnnotation; sortKey: number }
		| { kind: 'shot'; shot: ShotEventAnnotation; sortKey: number };

	interface Props {
		passEvents: PassEventAnnotation[];
		shotEvents: ShotEventAnnotation[];
		currentFrame: number;
		fps: number;
		onSeek: (time: number) => void;
	}

	let { passEvents, shotEvents, currentFrame, fps, onSeek }: Props = $props();

	function teamColor(teamId: number): string {
		return TEAM_COLORS[teamId] ?? TEAM_COLORS[0];
	}

	function formatTime(sec: number): string {
		const m = Math.floor(sec / 60);
		const s = Math.floor(sec % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}

	const rows = $derived.by((): Row[] => {
		const r: Row[] = [
			...passEvents.map((pass) => ({
				kind: 'pass' as const,
				pass,
				sortKey: pass.frame_start
			})),
			...shotEvents.map((shot) => ({
				kind: 'shot' as const,
				shot,
				sortKey: shot.frame_start
			}))
		];
		r.sort((a, b) => {
			const d = a.sortKey - b.sortKey;
			if (d !== 0) return d;
			const endA = a.kind === 'pass' ? a.pass.frame_end : a.shot.frame_end;
			const endB = b.kind === 'pass' ? b.pass.frame_end : b.shot.frame_end;
			const d2 = endA - endB;
			if (d2 !== 0) return d2;
			return a.kind === 'pass' ? -1 : 1;
		});
		return r;
	});

	let activeIndex = $derived.by(() => {
		const list = rows;
		for (let i = 0; i < list.length; i++) {
			const row = list[i];
			if (row.kind === 'pass') {
				const p = row.pass;
				if (p.frame_start <= currentFrame && p.frame_end >= currentFrame) return i;
			} else {
				const s = row.shot;
				if (s.frame_start <= currentFrame && s.frame_end >= currentFrame) return i;
			}
		}
		return -1;
	});
</script>

<div class="flex flex-col h-full min-h-0">
	<h3
		class="text-sm font-semibold text-[var(--color-text-muted)] px-4 py-3 border-b border-[var(--color-border)] flex items-center justify-between gap-2"
	>
		<span>Passes &amp; shots</span>
		<span class="text-xs font-normal tabular-nums text-[var(--color-text-muted)]">
			{passEvents.length} · {shotEvents.length}
		</span>
	</h3>

	<div class="flex-1 min-h-0 overflow-y-auto px-4 py-3 space-y-1.5">
		{#if passEvents.length === 0 && shotEvents.length === 0}
			<p class="text-xs text-[var(--color-text-muted)] text-center py-6">No passes or shots</p>
		{/if}
		{#each rows as row, i}
			{#if row.kind === 'pass'}
				{@const evt = row.pass}
				{@const isActive = i === activeIndex}
				<button
					onclick={() => onSeek((evt.frame_start + 0.5) / fps)}
					class="w-full text-left rounded-lg px-3 py-2 transition-all group
						{isActive
						? 'bg-[var(--color-ball)]/10 border border-[var(--color-ball)]/30'
						: 'hover:bg-[var(--color-surface-hover)] border border-transparent'}"
				>
					<div class="flex items-center gap-2 flex-wrap">
						<span
							class="text-[9px] font-semibold uppercase tracking-wide px-1.5 py-0.5 rounded bg-[var(--color-ball)]/15 text-[var(--color-ball)] border border-[var(--color-ball)]/25"
						>
							Pass
						</span>
						<span
							class="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded transition-colors
								{isActive
								? 'bg-[var(--color-ball)]/20 text-[var(--color-ball)]'
								: 'bg-[var(--color-accent)]/20 text-[var(--color-accent)] group-hover:bg-[var(--color-accent)]/30'}"
						>
							{formatTime(evt.timestamp_sec)}
						</span>
						<span class="text-[10px] text-[var(--color-text-muted)]"
							>f{evt.frame_start}–{evt.frame_end}</span
						>
						<div class="flex items-center gap-1.5 text-sm">
							<span
								class="inline-flex items-center gap-1 font-medium"
								style:color={teamColor(evt.team_id)}
							>
								<span
									class="w-1.5 h-1.5 rounded-full inline-block"
									style:background={teamColor(evt.team_id)}
								></span>
								{evt.from_track_number != null ? `#${evt.from_track_number} ` : ''}ID {evt.from_player_id}
							</span>
							<span class="text-[var(--color-text-muted)]">→</span>
							<span
								class="inline-flex items-center gap-1 font-medium"
								style:color={teamColor(evt.team_id)}
							>
								{evt.to_track_number != null ? `#${evt.to_track_number} ` : ''}ID {evt.to_player_id}
							</span>
						</div>
					</div>
				</button>
			{:else}
				{@const evt = row.shot}
				{@const isActive = i === activeIndex}
				<button
					onclick={() => onSeek(evt.timestamp_start_sec)}
					class="w-full text-left rounded-lg px-3 py-2 transition-all group
						{isActive
						? evt.is_make
							? 'bg-emerald-500/10 border border-emerald-400/35'
							: 'bg-amber-500/10 border border-amber-400/35'
						: 'hover:bg-[var(--color-surface-hover)] border border-transparent'}"
				>
					<div class="flex items-center gap-2 flex-wrap">
						<span
							class="text-[9px] font-semibold uppercase tracking-wide px-1.5 py-0.5 rounded border
								{evt.is_make
								? 'bg-emerald-500/18 text-emerald-300 border-emerald-400/30'
								: 'bg-amber-500/12 text-amber-200/95 border-amber-400/25'}"
						>
							Shot
						</span>
						<span
							class="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded transition-colors
								{isActive
								? evt.is_make
									? 'bg-emerald-500/25 text-emerald-300'
									: 'bg-amber-500/25 text-amber-200'
								: 'bg-[var(--color-accent)]/15 text-[var(--color-accent)] group-hover:bg-[var(--color-accent)]/25'}"
						>
							{formatTime(evt.timestamp_start_sec)} — {formatTime(evt.timestamp_end_sec)}
						</span>
						<span class="text-xs text-[var(--color-text-muted)]">
							f{evt.frame_start}–{evt.frame_end}
						</span>
						{#if evt.is_make}
							<span
								class="text-[10px] font-semibold uppercase tracking-wide px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-400/30"
							>
								Make
							</span>
						{/if}
						{#if evt.shooter_player_id != null}
							<span
								class="inline-flex items-center gap-1 text-sm font-medium"
								style:color={evt.shooter_team_id != null ? teamColor(evt.shooter_team_id) : 'var(--color-text)'}
							>
								<span
									class="w-1.5 h-1.5 rounded-full inline-block"
									style:background={evt.shooter_team_id != null ? teamColor(evt.shooter_team_id) : 'var(--color-text-muted)'}
								></span>
								{evt.shooter_track_number != null ? `#${evt.shooter_track_number} ` : ''}ID {evt.shooter_player_id}
							</span>
						{/if}
					</div>
				</button>
			{/if}
		{/each}
	</div>
</div>
