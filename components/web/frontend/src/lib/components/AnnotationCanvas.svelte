<script lang="ts">
	import type { AnnotationData, ToggleState } from '$lib/types';
	import { drawAnnotations } from '$lib/draw';

	interface Props {
		annotations: AnnotationData;
		frame: number;
		toggles: ToggleState;
		width: number;
		height: number;
	}

	let { annotations, frame, toggles, width, height }: Props = $props();

	let canvas: HTMLCanvasElement;

	$effect(() => {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const fd = annotations.frames[String(frame)];
		if (!fd) return;
		ctx.clearRect(0, 0, width, height);
		drawAnnotations(
			ctx,
			fd,
			toggles,
			width / annotations.metadata.width,
			height / annotations.metadata.height
		);
	});
</script>

<canvas
	bind:this={canvas}
	{width}
	{height}
	class="absolute top-0 left-0 pointer-events-none"
	style="width: {width}px; height: {height}px;"
></canvas>
