<script lang="ts">
	import type { AnnotationData, PlayerAnnotation } from '$lib/types';
	import { playerColor } from '$lib/draw';

	interface Props {
		annotations: AnnotationData | null;
		currentFrame: number;
		colorMode: 'team' | 'id';
	}

	let { annotations, currentFrame, colorMode }: Props = $props();

	let owner = $derived.by(() => {
		if (!annotations) return null;
		const frame = annotations.frames[String(currentFrame)];
		if (!frame) return null;
		return frame.players.find((p) => p.is_possession) ?? null;
	});

	function label(p: PlayerAnnotation): string {
		let parts: string[] = [];
		if (p.track_number != null) parts.push(`#${p.track_number}`);
		parts.push(`ID ${p.player_id}`);
		if (p.team_id != null) parts.push(`Team ${p.team_id + 1}`);
		return parts.join(' · ');
	}
</script>

<div
	class="flex items-center gap-3 px-4 py-2.5 rounded-xl border
	{owner
		? 'border-[var(--color-ball)]/40 bg-[var(--color-ball)]/5'
		: 'border-[var(--color-border)] bg-[var(--color-surface)]'}
	transition-all duration-200"
>
	<div class="flex items-center gap-2 shrink-0">
		<span class="text-sm">🏀</span>
		<span class="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
			Ball
		</span>
	</div>

	{#if owner}
		<div class="flex items-center gap-2">
			<span
				class="w-2.5 h-2.5 rounded-full shrink-0"
				style:background={playerColor(owner, colorMode)}
			></span>
			<span class="text-sm font-medium text-[var(--color-text)]">
				{label(owner)}
			</span>
		</div>
	{:else}
		<span class="text-sm text-[var(--color-text-muted)] italic">No possession</span>
	{/if}
</div>
