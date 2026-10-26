// Test browser-independent queue logic with mocked DOM; no real API/network.
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const path = require("node:path");
const elements = new Map();
const element = (id) => {
  if (!elements.has(id)) elements.set(id, {
    checked: false, disabled: false, value: "unmarked", textContent: "",
    addEventListener() {}, setAttribute() {},
  });
  return elements.get(id);
};
const context = vm.createContext({
  document: {getElementById: element, querySelectorAll: () => [], addEventListener() {}},
  window: {addEventListener() {}}, localStorage: {getItem: () => null, setItem() {}, removeItem() {}},
  location: {hash: ""}, setTimeout, clearTimeout, URL, console,
  fetch() { throw Error("TEST MUST NOT MAKE NETWORK REQUESTS"); },
});
const root = path.join(__dirname, "..");
const source = fs.readFileSync(path.join(root, "static/app.js"), "utf8").replace(/\nstart\(\);\s*$/, "");
vm.runInContext(source, context);
const run = (js) => vm.runInContext(js, context);
run("state.frames=[{output_frame:1,person_ids:[1,2]},{output_frame:2,person_ids:[1,2]}]; state.hardTargets=new Set(['1:2','2:1']);");
assert.equal(run("queuePairs().length"), 4);
element("hard-only").checked = true;
assert.equal(run("JSON.stringify(queuePairs())"), '[{"frame":1,"person":2},{"frame":2,"person":1}]');
assert.equal(run("frameComplete(state.frames[0])"), false);
run("state.annotations.set('1:2',{label:'sitting'});");
assert.equal(run("frameComplete(state.frames[0])"), true);
assert.equal(run("frameComplete(state.frames[1])"), false);
run("state.hardTargets=new Set(); lockUI(false);");
assert.equal(element("hard-only").disabled, true);
run("state.hardTargets=new Set(['1:2']); state.frame={output_frame:1,people:[{person_id:2}]}; state.personIndex=0;");
(async () => {
  await assert.rejects(run("save('sitting')"), /请先选择腿部可见程度/);
  assert.equal(run("state.annotations.get('1:2').label"), "sitting");
  const html = fs.readFileSync(path.join(root, "static/index.html"), "utf8");
  assert.equal(html.includes('id="hard-sample-data"'), false);
  run("state.config={dataset_id:'dataset-a'}; state.busy=false; applyHardQueue({status:'active', dataset_id:'dataset-a', batch:'synthetic', targets:[[2,2]]});");
  assert.equal(run("JSON.stringify([...state.hardTargets])"), '["2:2"]');
  assert.equal(element("hard-only").disabled, false);
  element("hard-only").checked = true;
  run("applyHardQueue({status:'active', dataset_id:'other', batch:'wrong', targets:[[1,1]]});");
  assert.equal(run("state.hardTargets.size"), 0);
  assert.equal(element("hard-only").checked, false);
  assert.equal(element("hard-only").disabled, true);
  run("applyHardQueue({status:'none', dataset_id:'dataset-a', targets:[]});");
  assert.equal(run("state.hardBatch"), null);
  console.log("PASS: filtering, target completion, API queue refresh, dataset guard, occlusion requirement, no embedded or historical manifest dependency.");
})().catch((error) => { console.error(error); process.exitCode = 1; });
