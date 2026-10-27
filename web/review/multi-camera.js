'use strict';
// The top camera owns the clock; wrist views follow without independent controls.
class MultiCamera {
  constructor(master, followers, onError) {
    this.master = master;
    this.followers = followers;
    this.onError = onError;
    this.generation = 0;
    for (const follower of followers) {
      follower.muted = true;
      follower.addEventListener('loadedmetadata', () => this.sync(true));
      follower.addEventListener('error', () => onError(new Error('腕相机无法加载：' + follower.getAttribute('src'))));
    }
    master.addEventListener('play', () => this.play());
    master.addEventListener('pause', () => { this.pause(); this.sync(true); });
    master.addEventListener('seeking', () => { this.pause(); this.sync(true); });
    master.addEventListener('seeked', () => { this.sync(true); if (!master.paused) this.play(); });
    master.addEventListener('ratechange', () => this.sync(false));
    master.addEventListener('waiting', () => this.pause());
    master.addEventListener('playing', () => this.play());
    master.addEventListener('ended', () => { this.pause(); this.sync(true); });
    const tick = () => { this.sync(false); master.requestVideoFrameCallback(tick); };
    master.requestVideoFrameCallback(tick);
  }
  load(sources) {
    if (!Array.isArray(sources) || sources.length !== this.followers.length || sources.some(src => typeof src !== 'string' || !src)) {
      throw new Error('必须指定两路腕相机视频地址');
    }
    this.generation++;
    this.pause();
    this.followers.forEach((video, index) => {
      video.src = sources[index];
      video.load();
    });
  }
  pause() { this.followers.forEach(video => video.pause()); }
  sync(force) {
    for (const video of this.followers) {
      video.playbackRate = this.master.playbackRate;
      if (video.readyState < 1) continue;
      if (force || Math.abs(video.currentTime - this.master.currentTime) > 1 / 30) {
        video.currentTime = Math.min(this.master.currentTime, Math.max(0, video.duration - 1 / 30));
      }
    }
  }
  play() {
    if (this.master.paused || this.master.seeking) return;
    this.sync(true);
    const generation = this.generation;
    for (const video of this.followers) {
      video.play().catch(error => {
        if (error.name === 'AbortError' && (generation !== this.generation || this.master.paused || this.master.seeking || video.paused)) return;
        this.master.pause();
        this.onError(error);
      });
    }
  }
}
