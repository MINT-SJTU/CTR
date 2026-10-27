const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor() { this.children = []; this.style = {}; this.attributes = {}; this.classList = {toggle() {}}; }
  replaceChildren() { this.children = []; }
  append(...elements) { this.children.push(...elements); }
  setAttribute(name, value) { this.attributes[name] = value; }
}
const context = vm.createContext({document: {createElement: () => new Element()}});
vm.runInContext(fs.readFileSync('web/review/idle-timeline.js', 'utf8') + '\nthis.Timeline = IdleTimeline;', context);
const timeline = new context.Timeline(new Element(), () => {});
const row = {output: 0, source: 2, frames: 5, fps: 30};
// Only diagnostic metadata is supplied: rendering must derive intervals from the new field.
const rest = {start: 4, mask_start: 4, supervised_frames: 0, max_supervised_frames: 45};
const labels = {...row, left_terminal_rest: rest, right_terminal_rest: rest};
const active = [[0, 1], [0, 0], [1, 0], [1, 1], [0, 1]];
timeline.setEpisode(row, labels, active);
assert.deepEqual(JSON.parse(JSON.stringify(timeline.tracks.map(t => t.intervals))), [[[0, 2], [4, 5]], [[1, 3]]]);
for (let frame = 0; frame < active.length; frame++) {
  timeline.update(frame);
  for (let side = 0; side < 2; side++) {
    assert.match(timeline.tracks[side].status.textContent, new RegExp(`mask=${active[frame][side]}`));
  }
}
assert.throws(() => timeline.setEpisode(row, labels, undefined));
assert.throws(() => timeline.setEpisode(row, labels, active.slice(1)));
assert.throws(() => timeline.setEpisode(row, labels, [[2, 1], ...active.slice(1)]));
console.log('Arm mask timeline: order, boundaries, statuses and invalid data passed');
for (const file of ['web/review/screw.html', 'web/review/workpiece.html']) {
 const html=fs.readFileSync(file,'utf8');
 for (const [, script] of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new vm.Script(script);
}
console.log('Preview page JavaScript syntax passed');
if (process.argv[2]) {
  const path = require('node:path');
  const root = process.argv[2];
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json')));
  const labels = JSON.parse(fs.readFileSync(path.join(root, 'idle-masks.json')));
  for (const row of manifest.episodes_data) {
    const label = labels.episodes.find(m => m.output === row.output);
    const data = JSON.parse(fs.readFileSync(path.join(root, `timelines/${String(row.output).padStart(3, '0')}.json`)));
    timeline.setEpisode(row, label, data['observation.arm_active_mask']);
    assert.deepEqual(JSON.parse(JSON.stringify(timeline.tracks.map(t => t.intervals))), [label.left, label.right]);
    for (let frame = 0; frame < row.frames; frame++) timeline.update(frame);
  }
  console.log(`Preview rendering passed for ${manifest.episodes_data.length} exported episodes`);
}
