// Confirm that the shipped Part 1 runtime reproduces every exported aabid row.
import fs from 'fs';
import vm from 'vm';

const toy = JSON.parse(fs.readFileSync(new URL('./toy1.json', import.meta.url), 'utf8'));
const source = fs.readFileSync(new URL('./part1.js', import.meta.url), 'utf8');
const AT = {
  matmul(a, B) {
    return B[0].map((_, j) => a.reduce((sum, value, i) => sum + value * B[i][j], 0));
  },
  softmax(values) {
    const max = Math.max(...values);
    const exp = values.map(value => Math.exp(value - max));
    const total = exp.reduce((a, b) => a + b, 0);
    return exp.map(value => value / total);
  },
  argmax(values) {
    return values.reduce((best, value, i) => value > values[best] ? i : best, 0);
  }
};
const context = { window: { AT, __TOY__: toy }, Set, Array, Math, Number };
vm.createContext(context);
vm.runInContext(source, context, { filename: 'part1.js' });

let worst = 0;
for (const row of toy.aabid_rows) {
  const got = context.window.AT.mlp.forward(row.ids).p;
  for (let i = 0; i < got.length; i++) worst = Math.max(worst, Math.abs(got[i] - row.probabilities[i]));
}
if (worst > 1e-6) {
  console.error(`Part 1 forward mismatch: max absolute difference ${worst}`);
  process.exit(1);
}
console.log(`Part 1 JS forward matches all six aabid rows; max absolute difference ${worst}.`);
