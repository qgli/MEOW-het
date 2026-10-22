const video = document.querySelector('#orbit-video');
const motionButton = document.querySelector('#motion-toggle');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let userPaused = reducedMotion.matches;
function showMotionState() {
  const paused = video.paused;
  motionButton.innerHTML = paused ? 'Play <span aria-hidden="true">▷</span>' : 'Pause <span aria-hidden="true">Ⅱ</span>';
  motionButton.setAttribute('aria-label', `${paused ? 'Play' : 'Pause'} reconstruction animation`);
}
if (userPaused) video.pause();
video.addEventListener('play', showMotionState);
video.addEventListener('pause', showMotionState);
motionButton.addEventListener('click', () => {
  userPaused = !video.paused;
  if (userPaused) video.pause();
  else video.play().catch(showMotionState);
  showMotionState();
});
reducedMotion.addEventListener('change', event => {
  userPaused = event.matches;
  if (userPaused) video.pause();
  else video.play().catch(showMotionState);
});
new IntersectionObserver(entries => {
  if (!entries[0].isIntersecting) video.pause();
  else if (!userPaused && !document.hidden) video.play().catch(showMotionState);
}, { threshold: 0.05 }).observe(video);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) video.pause();
  else if (!userPaused && video.getBoundingClientRect().bottom > 0) video.play().catch(showMotionState);
});
showMotionState();

const tabs = [...document.querySelectorAll('[data-tab]')];
function selectTab(selected) {
  for (const tab of tabs) {
    const active = tab === selected;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    document.querySelector(`#${tab.getAttribute('aria-controls')}`).hidden = !active;
  }
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 :
      (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    selectTab(tabs[next]);
    tabs[next].focus();
  });
});

const dialog = document.querySelector('#figure-dialog');
let zoomOrigin;
document.querySelectorAll('[data-zoom]').forEach(button => {
  button.addEventListener('click', () => {
    zoomOrigin = button;
    const image = document.querySelector('#expanded-figure');
    image.src = button.dataset.zoom;
    image.alt = button.querySelector('img').alt;
    document.querySelector('#expanded-caption').textContent = button.closest('figure').querySelector('figcaption')?.textContent || '';
    dialog.showModal();
  });
});
document.querySelector('#close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
dialog.addEventListener('close', () => zoomOrigin?.focus());

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  const citation = document.querySelector('#bibtex');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = 'BibTeX copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
