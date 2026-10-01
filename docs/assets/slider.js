// Homepage featured-story slider: native scroll-snap for swipe, buttons/dots for
// pointer and keyboard, gentle auto-advance that pauses on hover, focus or
// interaction and never runs for reduced-motion visitors.
(() => {
  document.querySelectorAll('.slider').forEach(root => {
    const track = root.querySelector('.slider-track');
    const slides = [...track.children];
    const dots = [...root.querySelectorAll('.slider-dots button')];
    if (slides.length < 2) return;
    let index = 0, paused = false, stopped = false;
    const go = i => {
      index = (i + slides.length) % slides.length;
      track.scrollTo({ left: slides[index].offsetLeft - track.offsetLeft, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
    };
    const mark = () => {
      index = Math.round(track.scrollLeft / track.clientWidth);
      dots.forEach((d, i) => d.setAttribute('aria-current', i === index ? 'true' : 'false'));
      slides.forEach((s, i) => { s.inert = i !== index; });
    };
    const stop = () => { stopped = true; };
    root.querySelector('.slider-prev').addEventListener('click', () => { stop(); go(index - 1); });
    root.querySelector('.slider-next').addEventListener('click', () => { stop(); go(index + 1); });
    dots.forEach((d, i) => d.addEventListener('click', () => { stop(); go(i); }));
    track.addEventListener('keydown', e => {
      if (e.key === 'ArrowRight') { stop(); go(index + 1); e.preventDefault(); }
      if (e.key === 'ArrowLeft') { stop(); go(index - 1); e.preventDefault(); }
    });
    track.addEventListener('scroll', () => requestAnimationFrame(mark), { passive: true });
    track.addEventListener('pointerdown', stop, { passive: true });
    root.addEventListener('mouseenter', () => { paused = true; });
    root.addEventListener('mouseleave', () => { paused = false; });
    root.addEventListener('focusin', () => { paused = true; });
    root.addEventListener('focusout', () => { paused = false; });
    mark();
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setInterval(() => { if (!paused && !stopped && !document.hidden) go(index + 1); }, 8000);
    }
  });
})();
