<script lang="ts">
	import type { ToggleState } from '$lib/types';

	interface Props {
		toggles: ToggleState;
	}

	let { toggles = $bindable() }: Props = $props();

	type ToggleKey = Exclude<keyof ToggleState, 'colorMode'>;
	const items: { key: ToggleKey; label: string; icon: string }[] = [
		{ key: 'bboxes', label: 'Bboxes', icon: '▢' },
		{ key: 'masks', label: 'Masks', icon: '◉' },
		{ key: 'skeletons', label: 'Skeletons', icon: '🦴' },
		{ key: 'ball', label: 'Ball', icon: '🏀' },
		{ key: 'trackNumbers', label: 'Track #', icon: '#' },
		{ key: 'rawJerseyNumbers', label: 'Raw #', icon: '⌗' },
		{ key: 'playerIds', label: 'IDs', icon: 'ID' },
		{ key: 'speedTrails', label: 'Trails', icon: '〰' },
		{ key: 'minimap', label: 'Minimap', icon: '🗺' },
		{ key: 'reidMatrix', label: 'ReID', icon: '⊞' }
	];

	function toggle(key: ToggleKey) {
		const next = { ...toggles, [key]: !toggles[key] } as ToggleState;
		// playerIds requires bboxes — turn it off when bboxes are hidden
		if (key === 'bboxes' && !next.bboxes) next.playerIds = false;
		toggles = next;
	}

	function isDisabled(key: ToggleKey): boolean {
		if (key === 'playerIds' && !toggles.bboxes) return true;
		return false;
	}
</script>

<div class="flex flex-wrap gap-2 items-center">
	{#each items as item}
		{@const disabled = isDisabled(item.key)}
		<button
			onclick={() => {
				if (!disabled) toggle(item.key);
			}}
			class="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all
				{disabled
				? 'opacity-40 cursor-not-allowed bg-[var(--color-surface)] text-[var(--color-text-muted)] border border-[var(--color-border)]'
				: toggles[item.key]
					? 'bg-[var(--color-accent)]/20 text-[var(--color-accent)] border border-[var(--color-accent)]/40'
					: 'bg-[var(--color-surface)] text-[var(--color-text-muted)] border border-[var(--color-border)] hover:border-[var(--color-text-muted)]'}"
		>
			<span class="text-sm">{item.icon}</span>
			{item.label}
		</button>
	{/each}

	<!-- Color mode segmented control -->
	<div
		class="flex items-center rounded-full border border-[var(--color-border)] overflow-hidden text-xs font-medium"
	>
		<button
			onclick={() => {
				toggles = { ...toggles, colorMode: 'team' };
			}}
			class="px-3 py-1.5 transition-all
				{toggles.colorMode === 'team'
				? 'bg-[var(--color-accent)]/20 text-[var(--color-accent)]'
				: 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'}"
		>
			Team
		</button>
		<span class="text-[var(--color-border)] select-none">|</span>
		<button
			onclick={() => {
				toggles = { ...toggles, colorMode: 'id' };
			}}
			class="px-3 py-1.5 transition-all
				{toggles.colorMode === 'id'
				? 'bg-[var(--color-accent)]/20 text-[var(--color-accent)]'
				: 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'}"
		>
			ID
		</button>
	</div>
</div>
