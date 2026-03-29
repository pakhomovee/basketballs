<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { getAnnotations, getJob, videoUrl } from '$lib/api';
	import type { AnnotationData, Job, ToggleState, ReidMatrix } from '$lib/types';
	import { DEFAULT_TOGGLES } from '$lib/types';
	import VideoPlayer from '$lib/components/VideoPlayer.svelte';
	import ControlPanel from '$lib/components/ControlPanel.svelte';
	import PossessionPanel from '$lib/components/PossessionPanel.svelte';
	import GameEventsPanel from '$lib/components/GameEventsPanel.svelte';
	import ReidMatrixPanel from '$lib/components/ReidMatrixPanel.svelte';
	import CommentsPanel from '$lib/components/CommentsPanel.svelte';

	let jobId = $derived($page.params.jobId);
	let job = $state<Job | null>(null);
	let annotations = $state<AnnotationData | null>(null);
	let jobLoading = $state(true);
	let error = $state('');
	let toggles = $state<ToggleState>({ ...DEFAULT_TOGGLES });
	let currentTime = $state(0);
	let currentFrame = $state(0);
	let courtTrails = $state<Record<number, [number, number][]>>({});
	let playerWidth = $state(0);
	let player = $state<VideoPlayer | undefined>(undefined);

	let reidMatrix = $derived<ReidMatrix | null>(
		annotations?.frames[String(currentFrame)]?.reid_cross_frame_matrix ?? null
	);

	onMount(async () => {
		if (!jobId) return;
		try {
			job = await getJob(jobId);
			if (job.status !== 'done') {
				error = `Video is ${job.status}. Please wait for processing to complete.`;
				return;
			}
		} catch (e: any) {
			error = e.message || 'Failed to load video';
		} finally {
			jobLoading = false;
		}
		// Load annotations in the background — video is already visible
		try {
			annotations = await getAnnotations(jobId!);
		} catch (e: any) {
			// Annotations failing shouldn't hide the video — just log
			console.error('Failed to load annotations:', e);
		}
	});

	function handleSeek(time: number) {
		player?.seekTo(time);
	}
</script>

<svelte:head>
	<title>{job?.video_name ?? 'Loading…'} — Basketball Analytics</title>
</svelte:head>

{#if jobLoading}
	<div class="flex items-center justify-center h-[60vh]">
		<div
			class="w-8 h-8 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin"
		></div>
	</div>
{:else if error}
	<div class="flex items-center justify-center h-[60vh]">
		<div class="text-center">
			<p class="text-lg text-red-400">{error}</p>
			<a href="/" class="text-sm text-[var(--color-accent)] mt-4 inline-block hover:underline"
				>← Back to videos</a
			>
		</div>
	</div>
{:else if job}
	<div class="flex h-[calc(100vh-65px)]">
		<!-- Main content -->
		<div
			class="shrink-0 flex flex-col min-w-0 p-4 gap-4 overflow-y-auto"
			style:width={playerWidth > 0 ? `${playerWidth + 32}px` : '75%'}
		>
			<!-- Video player with minimap overlay -->
			<div class="relative w-full">
				<VideoPlayer
					bind:this={player}
					src={videoUrl(jobId!)}
					{annotations}
					{toggles}
					bind:currentTime
					bind:currentFrame
					bind:courtTrails
					bind:playerWidth
				/>
			</div>

			<!-- Controls -->
			<ControlPanel bind:toggles />
		</div>

		<!-- Sidebar -->
		<div
			class="flex-1 min-w-0 flex flex-col border-l border-[var(--color-border)] bg-[var(--color-surface)]"
		>
			<!-- Possession indicator -->
			<div class="px-3 py-2.5 border-b border-[var(--color-border)]">
				<PossessionPanel {annotations} {currentFrame} colorMode={toggles.colorMode} />
			</div>
			<!-- Comments -->
			<div class="basis-[28%] min-h-0 shrink-0 border-b border-[var(--color-border)]">
				<CommentsPanel jobId={jobId!} {currentTime} onSeek={handleSeek} />
			</div>
			<!-- Passes & shots (single scroll list) -->
			<div class="flex-1 min-h-0 shrink-0 border-b border-[var(--color-border)] flex flex-col">
				<GameEventsPanel
					passEvents={annotations?.pass_events ?? []}
					shotEvents={annotations?.shot_events ?? []}
					{currentFrame}
					onSeek={handleSeek}
				/>
			</div>
			<!-- ReID matrix (scrollable remainder) -->
			<div class="flex-1 min-h-0 overflow-y-auto px-3 py-2">
				<ReidMatrixPanel matrix={reidMatrix} visible={toggles.reidMatrix} />
			</div>
		</div>
	</div>
{/if}
