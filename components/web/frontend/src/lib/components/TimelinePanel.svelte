<script lang="ts">
	import { addComment, getComments } from '$lib/api';
	import type { Comment, PassEventAnnotation, ShotEventAnnotation } from '$lib/types';
	import { onMount } from 'svelte';

	const TEAM_COLORS = ['#ff6b2b', '#3b82f6'];

	type Row =
		| { kind: 'pass'; pass: PassEventAnnotation; sortKey: number }
		| { kind: 'shot'; shot: ShotEventAnnotation; sortKey: number }
		| { kind: 'comment'; comment: Comment; sortKey: number };

	interface Props {
		jobId: string;
		passEvents: PassEventAnnotation[];
		shotEvents: ShotEventAnnotation[];
		currentFrame: number;
		currentTime: number;
		fps: number;
		onSeek: (time: number) => void;
	}

	let { jobId, passEvents, shotEvents, currentFrame, currentTime, fps, onSeek }: Props = $props();

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
			})),
			...comments.map((comment) => ({
				kind: 'comment' as const,
				comment,
				sortKey: Math.round(comment.timestamp_sec * fps)
			}))
		];
		r.sort((a, b) => {
			const d = a.sortKey - b.sortKey;
			if (d !== 0) return d;
			// within same frame: comments first, then passes, then shots
			const order = { comment: 0, pass: 1, shot: 2 };
			return order[a.kind] - order[b.kind];
		});
		return r;
	});

	function activeEventIndex(): number {
		const list = rows;
		for (let i = 0; i < list.length; i++) {
			const row = list[i];
			if (row.kind === 'pass') {
				const p = row.pass;
				if (p.frame_start <= currentFrame && p.frame_end >= currentFrame) return i;
			} else if (row.kind === 'shot') {
				const s = row.shot;
				if (s.frame_start <= currentFrame && s.frame_end >= currentFrame) return i;
			}
		}
		return -1;
	}

	let activeIndex = $derived(activeEventIndex());

	let eventCounts = $derived({
		passes: passEvents.length,
		shots: shotEvents.length,
		comments: comments.length
	});
</script>

<div class="flex flex-col h-full min-h-0">
	<h3
		class="text-sm font-semibold text-[var(--color-text-muted)] px-4 py-3 border-b border-[var(--color-border)] flex items-center justify-between gap-2"
	>
		<span>Timeline</span>
		<span class="text-xs font-normal tabular-nums text-[var(--color-text-muted)]">
			{eventCounts.passes}P · {eventCounts.shots}S · {eventCounts.comments}C
		</span>
	</h3>

	<!-- Scrollable timeline -->
	<div class="flex-1 min-h-0 overflow-y-auto px-4 py-3 space-y-1.5">
		{#if rows.length === 0}
			<p class="text-xs text-[var(--color-text-muted)] text-center py-6">No events yet</p>
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
			{:else if row.kind === 'shot'}
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
								style:color={evt.shooter_team_id != null
									? teamColor(evt.shooter_team_id)
									: 'var(--color-text)'}
							>
								<span
									class="w-1.5 h-1.5 rounded-full inline-block"
									style:background={evt.shooter_team_id != null
										? teamColor(evt.shooter_team_id)
										: 'var(--color-text-muted)'}
								></span>
								{evt.shooter_track_number != null ? `#${evt.shooter_track_number} ` : ''}ID {evt.shooter_player_id}
							</span>
						{/if}
					</div>
				</button>
			{:else}
				{@const c = row.comment}
				<button
					onclick={() => onSeek(c.timestamp_sec)}
					class="w-full text-left rounded-lg px-3 py-2 transition-all group hover:bg-[var(--color-surface-hover)] border border-transparent"
				>
					<div class="flex items-start gap-2">
						<span
							class="text-[9px] font-semibold uppercase tracking-wide px-1.5 py-0.5 rounded bg-[var(--color-accent)]/15 text-[var(--color-accent)] border border-[var(--color-accent)]/25 shrink-0 mt-0.5"
						>
							Note
						</span>
						<span
							class="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--color-accent)]/20 text-[var(--color-accent)] group-hover:bg-[var(--color-accent)]/30 transition-colors mt-0.5"
						>
							{formatTime(c.timestamp_sec)}
						</span>
						<p class="text-sm text-[var(--color-text)] leading-snug">{c.text}</p>
					</div>
				</button>
			{/if}
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
			Add at {formatTime(currentTime)}
		</button>
	</div>
</div>
