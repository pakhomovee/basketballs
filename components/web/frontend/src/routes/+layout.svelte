<script>
	import '../app.css';
	let { children } = $props();
</script>

<div class="min-h-screen bg-[var(--color-bg)]">
	<nav class="border-b border-[var(--color-border)] px-6 py-4">
		<a href="/" class="flex items-center gap-3 w-fit">
			<div
				class="w-8 h-8 rounded-full bg-[var(--color-accent)] flex items-center justify-center text-white font-bold text-sm"
			>
				B
			</div>
			<span class="text-lg font-semibold text-[var(--color-text)]">Basketball Analytics</span>
		</a>
	</nav>
	{@render children()}
</div>
