/** Landing page: render the clip gallery from data/manifest.json. */

async function loadManifest() {
	const res = await fetch('data/manifest.json');
	if (!res.ok) throw new Error(`manifest ${res.status}`);
	return res.json();
}

function card(v) {
	const available = v.available === true;
	const el = document.createElement(available ? 'a' : 'div');
	el.className = `card ${available ? 'available' : 'soon'}`;
	if (available) {
		el.href = `player.html?v=${encodeURIComponent(v.id)}`;
		el.classList.add('card-link');
	}

	const thumb = document.createElement('div');
	thumb.className = 'thumb';
	if (available) {
		// Muted, non-playing preview frame of the clip.
		const vid = document.createElement('video');
		vid.src = v.video;
		vid.muted = true;
		vid.preload = 'metadata';
		vid.playsInline = true;
		thumb.appendChild(vid);
	} else {
		thumb.textContent = '🏀';
		thumb.style.fontSize = '32px';
	}
	const badge = document.createElement('span');
	badge.className = `badge ${available ? 'live' : 'soon'}`;
	badge.textContent = available ? 'Ready' : 'Coming soon';
	thumb.appendChild(badge);

	const body = document.createElement('div');
	body.className = 'body';
	const h = document.createElement('h3');
	h.textContent = v.title;
	const p = document.createElement('p');
	p.textContent = v.description;
	const cta = document.createElement('div');
	cta.className = 'cta';
	cta.textContent = available ? 'Open viewer →' : 'Annotations pending';
	body.append(h, p, cta);

	el.append(thumb, body);
	return el;
}

async function main() {
	const gallery = document.getElementById('gallery');
	try {
		const { videos } = await loadManifest();
		gallery.replaceChildren(...videos.map(card));
	} catch (e) {
		gallery.innerHTML = `<p style="color:var(--text-muted)">Could not load clip list: ${e.message}</p>`;
	}
}

main();
