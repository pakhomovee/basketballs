<script lang="ts">
	import { addComment, getComments } from '$lib/api';
	import type { Comment } from '$lib/types';
	import { onMount } from 'svelte';

	interface Props {
		jobId: string;
		currentTime: number;
		onSeek: (time: number) => void;
	}

	let { jobId, currentTime, onSeek }: Props = $props();

	let comments = $state<Comment[]>([]);
	let newText = $state('');
	let submitting = $state(false);

	onMount(async () => {
		try {
			comments = await getComments(jobId);
		} catch {
			// Non-critical — leave list empty
		}
	});

	async function submitComment() {
		if (!newText.trim()) return;
		submitting = true;
		try {
			const comment = await addComment(jobId, currentTime, newText.trim());
			comments = [...comments, comment].sort((a, b) => a.timestamp_sec - b.timestamp_sec);
			newText = '';
		} catch {
			// silently fail
		} finally {
			submitting = false;
		}
	}

	function formatTimestamp(sec: number): string {
		const m = Math.floor(sec / 60);
		const s = Math.floor(sec % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}
</script>

<div class="flex flex-col h-full min-h-0">
	<h3
		class="text-sm font-semibold text-[var(--color-text-muted)] px-4 py-3 border-b border-[var(--color-border)]"
	>
		Comments
	</h3>

	<!-- Comment list -->
	<div class="flex-1 min-h-0 overflow-y-auto px-4 py-3 space-y-3">
		{#if comments.length === 0}
			<p class="text-xs text-[var(--color-text-muted)] text-center py-8">No comments yet</p>
		{/if}
		{#each comments as comment}
			<button onclick={() => onSeek(comment.timestamp_sec)} class="w-full text-left group">
				<div class="flex items-start gap-2">
					<span
						class="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--color-accent)]/20 text-[var(--color-accent)] group-hover:bg-[var(--color-accent)]/30 transition-colors"
					>
						{formatTimestamp(comment.timestamp_sec)}
					</span>
					<p class="text-sm text-[var(--color-text)] leading-snug">{comment.text}</p>
				</div>
			</button>
		{/each}
	</div>

	<!-- Add comment -->
	<div class="border-t border-[var(--color-border)] p-4">
		<div class="flex gap-2">
			<textarea
				bind:value={newText}
				placeholder="Add a comment…"
				rows="2"
				class="flex-1 bg-[var(--color-surface)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm text-[var(--color-text)] placeholder-[var(--color-text-muted)] resize-none focus:outline-none focus:border-[var(--color-accent)]"
				onkeydown={(e) => {
					if (e.key === 'Enter' && !e.shiftKey) {
						e.preventDefault();
						submitComment();
					}
				}}
			></textarea>
		</div>
		<button
			onclick={submitComment}
			disabled={submitting || !newText.trim()}
			class="mt-2 w-full px-3 py-2 rounded-lg text-sm font-medium transition-colors
				{newText.trim()
				? 'bg-[var(--color-accent)] text-white hover:opacity-90'
				: 'bg-[var(--color-surface)] text-[var(--color-text-muted)] cursor-not-allowed'}"
		>
			Add at {formatTimestamp(currentTime)}
		</button>
	</div>
</div>
