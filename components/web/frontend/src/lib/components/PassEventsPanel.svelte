<script lang="ts">
	import type { PassEventAnnotation } from '$lib/types';

	const TEAM_COLORS = ['#ff6b2b', '#3b82f6'];

	interface Props {
		passEvents: PassEventAnnotation[];
		currentFrame: number;
		onSeek: (time: number) => void;
	}

	let { passEvents, currentFrame, onSeek }: Props = $props();

	function teamColor(teamId: number): string {
		return TEAM_COLORS[teamId] ?? TEAM_COLORS[0];
	}

	function formatTime(sec: number): string {
		const m = Math.floor(sec / 60);
		const s = Math.floor(sec % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}

	let activeIdx = $derived.by(() => {
		if (passEvents.length === 0) return -1;
		let best = -1;
		for (let i = 0; i < passEvents.length; i++) {
			if (passEvents[i].from_frame <= currentFrame && passEvents[i].frame >= currentFrame) return i;
			if (passEvents[i].frame <= currentFrame) best = i;
		}
		return best;
	});
</script>

<div class="flex flex-col h-full min-h-0">
	<h3
		class="text-sm font-semibold text-[var(--color-text-muted)] px-4 py-3 border-b border-[var(--color-border)] flex items-center justify-between"
	>
		<span>Passes</span>
		<span class="text-xs font-normal tabular-nums">{passEvents.length}</span>
	</h3>

	<div class="flex-1 min-h-0 overflow-y-auto px-4 py-3 space-y-1.5">
		{#if passEvents.length === 0}
			<p class="text-xs text-[var(--color-text-muted)] text-center py-6">No passes detected</p>
		{/if}
		{#each passEvents as evt, i}
			{@const isActive = i === activeIdx}
			<button
				onclick={() => onSeek(evt.timestamp_sec)}
				class="w-full text-left rounded-lg px-3 py-2 transition-all group
					{isActive
					? 'bg-[var(--color-ball)]/10 border border-[var(--color-ball)]/30'
					: 'hover:bg-[var(--color-surface-hover)] border border-transparent'}"
			>
				<div class="flex items-center gap-2">
					<span
						class="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded transition-colors
							{isActive
							? 'bg-[var(--color-ball)]/20 text-[var(--color-ball)]'
							: 'bg-[var(--color-accent)]/20 text-[var(--color-accent)] group-hover:bg-[var(--color-accent)]/30'}"
					>
						{formatTime(evt.timestamp_sec)}
					</span>

					<div class="flex items-center gap-1.5 text-sm">
						<span
							class="inline-flex items-center gap-1 font-medium"
							style:color={teamColor(evt.team_id)}
						>
							<span
								class="w-1.5 h-1.5 rounded-full inline-block"
								style:background={teamColor(evt.team_id)}
							></span>
							#{evt.from_player_id}
						</span>
						<span class="text-[var(--color-text-muted)]">→</span>
						<span
							class="inline-flex items-center gap-1 font-medium"
							style:color={teamColor(evt.team_id)}
						>
							#{evt.to_player_id}
						</span>
					</div>
				</div>
			</button>
		{/each}
	</div>
</div>
