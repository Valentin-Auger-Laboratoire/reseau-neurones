// Tests du réseau de neurones (node tests/test.js)
'use strict';
const N = require('../nn.js');
const DIG = N.parseDigits(require('../data/chiffres.js'));
let n = 0, ko = 0; const fails = [];
function ok(c, m) { n++; if (!c) { ko++; fails.push(m); } }
const near = (a, b, t) => Math.abs(a - b) <= t;
const R = N.rng(2026), U = (a, b) => a + (b - a) * R();

// ------------------------------------------------------------ 1. rétropropagation = différences finies
for (const act of ['relu', 'leaky', 'tanh', 'sigmoid', 'linear']) for (const out of ['sigmoid', 'softmax', 'linear']) for (const depth of [0, 1, 3]) for (const lambda of [0, 0.02]) {
  const K = out === 'softmax' ? 4 : out === 'linear' ? 2 : 1, sizes = [3].concat(Array(depth).fill(5), [K]);
  const net = N.create(sizes, act, out, 7 + depth);
  // ReLU : on tire des exemples loin des points anguleux (z = 0), où la dérivée n'existe pas
  let X; for (let k = 0; k < 500; k++) { X = Array.from({ length: 7 }, () => [U(-1, 1), U(-1, 1), U(-1, 1)]); if (!(act === 'relu' || act === 'leaky') || N.minPreact(net, X) > 1e-3) break; }
  ok(!(act === 'relu' || act === 'leaky') || N.minPreact(net, X) > 1e-3, 'exemples loin des points anguleux (' + act + ', ' + depth + ')');
  const Y = X.map((x, i) => (out === 'softmax' ? i % K : out === 'sigmoid' ? i % 2 : [x[0] - x[1], x[2] * 2]));
  const { worst, absMax, count } = N.gradCheck(net, X, Y, lambda, 1e-5);
  ok(absMax < 1e-8, 'gradient ' + act + ' → ' + out + ', ' + depth + ' : écart absolu ' + absMax.toExponential(1));
  ok(worst < 1e-6, 'gradient ' + act + ' → ' + out + ', ' + depth + ' couche(s) cachée(s), L2 ' + lambda + ' : erreur relative ' + worst.toExponential(1) + ' sur ' + count + ' paramètres');
}
// dérivées des activations contre la dérivée numérique
for (const [name, A] of Object.entries(N.ACT)) for (let k = 0; k < 40; k++) {
  let z = U(-4, 4); if (Math.abs(z) < 1e-3) z = 0.5;
  const h = 1e-6, num = (A.f(z + h) - A.f(z - h)) / (2 * h);
  ok(near(A.d(z, A.f(z)), num, 1e-6), 'dérivée de ' + name + ' en z = ' + z.toFixed(3));
}

// ------------------------------------------------------------ 2. softmax, pertes
{
  const p = N.softmax(Float64Array.from([1000, 1001, 999]));
  ok(near(p[0] + p[1] + p[2], 1, 1e-12) && p.every((v) => isFinite(v)), 'softmax stable avec des entrées de 1 000');
  ok(near(p[1], Math.exp(1) / (1 + Math.exp(1) + Math.exp(-1)), 1e-12), 'softmax : valeur exacte');
  const net = N.create([2, 3], 'relu', 'softmax', 1);
  ok(near(N.lossOf(net, Float64Array.from([0.2, 0.5, 0.3]), 1), -Math.log(0.5), 1e-12), 'entropie croisée = −ln p(classe)');
  const nb = N.create([2, 1], 'relu', 'sigmoid', 1);
  ok(near(N.lossOf(nb, Float64Array.from([0.8]), 0), -Math.log(0.2), 1e-12), 'entropie croisée binaire');
  ok(near(N.lossOf(nb, Float64Array.from([1]), 1), 0, 1e-9) && isFinite(N.lossOf(nb, Float64Array.from([1]), 0)), 'entropie croisée bornée quand p = 1');
}

// ------------------------------------------------------------ 3. initialisation
{
  const net = N.create([400, 400, 1], 'relu', 'sigmoid', 3), w = net.W[0];
  const sd = Math.sqrt(w.reduce((s, v) => s + v * v, 0) / w.length);
  ok(near(sd / Math.sqrt(2 / 400), 1, 0.02), 'He : écart-type √(2/n) pour ReLU (' + (sd / Math.sqrt(2 / 400)).toFixed(3) + ')');
  const nt = N.create([400, 400, 1], 'tanh', 'sigmoid', 3), wt = nt.W[0], sdt = Math.sqrt(wt.reduce((s, v) => s + v * v, 0) / wt.length);
  ok(near(sdt / Math.sqrt(2 / 800), 1, 0.02), 'Glorot : écart-type √(2/(n_in + n_out)) pour tanh');
  ok(net.b.every((b) => b.every((v) => v === 0)), 'biais initialisés à zéro');
  const a = N.create([5, 7, 3], 'tanh', 'softmax', 42), b = N.create([5, 7, 3], 'tanh', 'softmax', 42);
  ok(a.W.every((w, l) => w.every((v, k) => v === b.W[l][k])), 'même graine → mêmes poids');
  ok(N.nParams(N.create([2, 8, 8, 1], 'tanh', 'sigmoid', 1)) === 2 * 8 + 8 + 8 * 8 + 8 + 8 + 1, 'nombre de paramètres');
}

// ------------------------------------------------------------ 4. optimiseurs
{
  // Adam, premier pas : Δ = −lr · g / (|g| + ε) grâce à la correction de biais
  const net = N.create([2, 1], 'relu', 'linear', 1), w0 = Float64Array.from(net.W[0]), g = { W: [Float64Array.from([0.3, -2])], b: [Float64Array.from([1e-3])] };
  const o = N.optimizer(net, { type: 'adam', lr: 0.1 }); o.step(g);
  ok(near(net.W[0][0] - w0[0], -0.1 * 0.3 / (0.3 + 1e-8), 1e-12) && near(net.W[0][1] - w0[1], 0.1, 1e-9), 'Adam : premier pas = lr × signe du gradient');
  // moment : gradient constant → vitesse g(1 − β^t)/(1 − β)
  const n2 = N.create([1, 1], 'relu', 'linear', 1), o2 = N.optimizer(n2, { type: 'momentum', lr: 0.01, beta: 0.9 }), start = n2.W[0][0];
  const gc = { W: [Float64Array.from([1])], b: [Float64Array.from([0])] };
  for (let t = 0; t < 20; t++) o2.step(gc);
  let expect = 0; for (let t = 1; t <= 20; t++) expect += (1 - 0.9 ** t) / (1 - 0.9);
  ok(near(start - n2.W[0][0], 0.01 * expect, 1e-12), 'moment : somme géométrique exacte');
  // RMSprop : premier pas = lr·g/(√((1−ρ)g²) + ε)
  const n3 = N.create([1, 1], 'relu', 'linear', 1), o3 = N.optimizer(n3, { type: 'rmsprop', lr: 0.01 }), s3 = n3.W[0][0];
  o3.step({ W: [Float64Array.from([2])], b: [Float64Array.from([0])] });
  ok(near(s3 - n3.W[0][0], (0.01 * 2) / (Math.sqrt(0.1 * 4) + 1e-8), 1e-12), 'RMSprop : premier pas exact');
}
{
  // régression linéaire : la descente de gradient retrouve les moindres carrés
  const X = [], Y = []; for (let i = 0; i < 60; i++) { const a = U(-1, 1), b = U(-1, 1); X.push([a, b]); Y.push([1.5 * a - 0.7 * b + 0.3 + 0.05 * N.gauss(R)]); }
  // solution exacte (équations normales 3×3)
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], v = [0, 0, 0];
  X.forEach((x, i) => { const f = [x[0], x[1], 1]; for (let p = 0; p < 3; p++) { v[p] += f[p] * Y[i][0]; for (let q = 0; q < 3; q++) A[p][q] += f[p] * f[q]; } });
  for (let p = 0; p < 3; p++) for (let r = p + 1; r < 3; r++) { const m = A[r][p] / A[p][p]; for (let q = p; q < 3; q++) A[r][q] -= m * A[p][q]; v[r] -= m * v[p]; }
  const s = [0, 0, 0]; for (let p = 2; p >= 0; p--) { let t = v[p]; for (let q = p + 1; q < 3; q++) t -= A[p][q] * s[q]; s[p] = t / A[p][p]; }
  const net = N.create([2, 1], 'relu', 'linear', 1), o = N.optimizer(net, { type: 'sgd', lr: 0.5 });
  for (let k = 0; k < 3000; k++) o.step(N.batchGrad(net, X, Y).g);
  ok(near(net.W[0][0], s[0], 1e-6) && near(net.W[0][1], s[1], 1e-6) && near(net.b[0][0], s[2], 1e-6), 'régression linéaire : descente de gradient = moindres carrés à 10⁻⁶');
}
{
  // pas trop grand : la perte explose
  const d = N.dataset('circle', 100, 0.1, 1), net = N.create([2, 8, 1], 'relu', 'sigmoid', 1), o = N.optimizer(net, { type: 'sgd', lr: 200 });
  const l0 = N.meanLoss(net, d.P, d.Y); for (let e = 0; e < 30; e++) N.epoch(net, o, d.P, d.Y, 10, 0, N.rng(2));
  const l1 = N.meanLoss(net, d.P, d.Y);
  ok(!(l1 < l0 * 0.5), 'pas d\'apprentissage de 200 : pas de convergence (perte ' + l0.toFixed(2) + ' → ' + (isFinite(l1) ? l1.toFixed(2) : l1) + ')');
}

// ------------------------------------------------------------ 5. ce qu'un réseau peut ou ne peut pas apprendre
{
  const d = N.dataset('xor', 200, 0, 4);
  const lin = N.create([2, 1], 'tanh', 'sigmoid', 1), ol = N.optimizer(lin, { type: 'adam', lr: 0.05 });
  for (let e = 0; e < 200; e++) N.epoch(lin, ol, d.P, d.Y, 20, 0, N.rng(1));
  ok(N.accuracy(lin, d.P, d.Y) < 0.8, 'XOR sans couche cachée : impossible (' + (N.accuracy(lin, d.P, d.Y) * 100).toFixed(0) + ' %)');
  let solved = 0;
  for (const seed of [1, 2, 3, 4, 5]) { const net = N.create([2, 4, 1], 'tanh', 'sigmoid', seed), o = N.optimizer(net, { type: 'adam', lr: 0.03 }); for (let e = 0; e < 300; e++) N.epoch(net, o, d.P, d.Y, 20, 0, N.rng(seed)); if (N.accuracy(net, d.P, d.Y) >= 0.98) solved++; }
  ok(solved >= 4, 'XOR avec 4 neurones cachés : résolu (' + solved + ' graines sur 5)');
  const Xf = d.P.map((p) => N.featurize(p, ['x', 'y', 'xy'])), lf = N.create([3, 1], 'tanh', 'sigmoid', 1), of = N.optimizer(lf, { type: 'adam', lr: 0.05 });
  for (let e = 0; e < 200; e++) N.epoch(lf, of, Xf, d.Y, 20, 0, N.rng(1));
  ok(N.accuracy(lf, Xf, d.Y) >= 0.98, 'XOR sans couche cachée mais avec x·y en entrée : résolu');
}
for (const kind of ['circle', 'moons', 'gauss']) {
  const d = N.dataset(kind, 300, 0.1, 5), net = N.create([2, 8, 1], 'tanh', 'sigmoid', 1), o = N.optimizer(net, { type: 'adam', lr: 0.03 });
  for (let e = 0; e < 200; e++) N.epoch(net, o, d.P, d.Y, 16, 0, N.rng(1));
  ok(N.accuracy(net, d.P, d.Y) > 0.95, kind + ' : appris (' + (N.accuracy(net, d.P, d.Y) * 100).toFixed(1) + ' %)');
}
{
  const d = N.dataset('spiral', 400, 0.05, 5), net = N.create([2, 16, 16, 16, 1], 'tanh', 'sigmoid', 2), o = N.optimizer(net, { type: 'adam', lr: 0.01 });
  for (let e = 0; e < 400; e++) N.epoch(net, o, d.P, d.Y, 16, 0, N.rng(1));
  ok(N.accuracy(net, d.P, d.Y) > 0.95, 'spirale : apprise par 3 couches de 16 (' + (N.accuracy(net, d.P, d.Y) * 100).toFixed(1) + ' %)');
}
{
  // surapprentissage : peu de points, gros réseau ; L2 réduit l'écart apprentissage / test
  const d = N.dataset('moons', 500, 2, 8), sp = N.split(500, 0.9, 3);
  const Xtr = sp.train.map((i) => d.P[i]), Ytr = sp.train.map((i) => d.Y[i]), Xte = sp.test.map((i) => d.P[i]), Yte = sp.test.map((i) => d.Y[i]);
  const run = (lambda) => { const net = N.create([2, 32, 32, 1], 'relu', 'sigmoid', 4), o = N.optimizer(net, { type: 'adam', lr: 0.01 }); for (let e = 0; e < 600; e++) N.epoch(net, o, Xtr, Ytr, 10, lambda, N.rng(2)); return [N.meanLoss(net, Xtr, Ytr), N.meanLoss(net, Xte, Yte)]; };
  const [tr0, te0] = run(0), [tr1, te1] = run(0.01);
  ok(te0 > 2 * tr0, 'surapprentissage : perte de test bien plus haute que celle d\'apprentissage (' + tr0.toFixed(3) + ' / ' + te0.toFixed(3) + ')');
  ok(te1 < te0, 'L2 : la perte de test baisse (' + te0.toFixed(3) + ' → ' + te1.toFixed(3) + ')');
}

// ------------------------------------------------------------ 6. jeux de données
for (const kind of N.DATASETS) {
  const d = N.dataset(kind, 600, 0, 9), counts = new Array(d.K).fill(0); d.Y.forEach((c) => counts[c]++);
  ok(counts.every((c) => Math.abs(c - 600 / d.K) <= 0.15 * 600 / d.K), kind + ' : classes équilibrées (' + counts.join(' / ') + ')');
  if (kind !== 'gauss') ok(d.P.every(([x, y]) => Math.abs(x) <= 1 && Math.abs(y) <= 1), kind + ' : points dans [−1, 1]² sans bruit');
  const a = N.dataset(kind, 50, 0.3, 9), b = N.dataset(kind, 50, 0.3, 9); ok(a.P.every((p, i) => p[0] === b.P[i][0] && p[1] === b.P[i][1]), kind + ' : reproductible');
}
{
  const c = N.dataset('circle', 400, 0, 1); ok(c.P.every(([x, y], i) => (Math.hypot(x, y) < 0.5) === (c.Y[i] === 1)), 'cercles : classe = intérieur');
  const x = N.dataset('xor', 400, 0, 1); ok(x.P.every(([a, b], i) => (a * b > 0 ? 0 : 1) === x.Y[i]), 'XOR : classe = signe de x·y');
  const sp = N.split(100, 0.3, 4), all = sp.train.concat(sp.test).sort((a, b) => a - b);
  ok(sp.test.length === 30 && all.every((v, i) => v === i), 'découpage apprentissage / test : disjoint et complet');
}

// ------------------------------------------------------------ 7. chiffres manuscrits
{
  ok(DIG.X.length === 1797 && DIG.X.every((x) => x.length === 64 && x.every((v) => v >= 0 && v <= 1)), '1 797 chiffres de 64 pixels');
  const counts = new Array(10).fill(0); DIG.Y.forEach((c) => counts[c]++); ok(counts.every((c) => c >= 170 && c <= 185), 'environ 180 exemples par chiffre');
  const full = new Uint8Array(1024).fill(1); ok(N.blocks32to8(full).every((v) => v === 1), 'blocs 4×4 : image pleine → 16/16 partout');
  const one = new Uint8Array(1024); one[0] = 1; one[33] = 1; const b = N.blocks32to8(one); ok(near(b[0], 2 / 16, 1e-12) && b.slice(1).every((v) => v === 0), 'blocs 4×4 : comptage des pixels');
  const sp = N.split(DIG.X.length, 0.2, 11);
  const Xtr = sp.train.map((i) => DIG.X[i]), Ytr = sp.train.map((i) => DIG.Y[i]), Xte = sp.test.map((i) => DIG.X[i]), Yte = sp.test.map((i) => DIG.Y[i]);
  const net = N.create([64, 32, 10], 'relu', 'softmax', 1), o = N.optimizer(net, { type: 'adam', lr: 0.005 }), r = N.rng(4);
  for (let e = 0; e < 30; e++) N.epoch(net, o, Xtr, Ytr, 32, 1e-4, r);
  const atr = N.accuracy(net, Xtr, Ytr), ate = N.accuracy(net, Xte, Yte);
  ok(atr > 0.99, 'chiffres : ' + (atr * 100).toFixed(1) + ' % sur l\'apprentissage');
  ok(ate > 0.94, 'chiffres : ' + (ate * 100).toFixed(1) + ' % sur 359 chiffres jamais vus');
  const M = N.confusion(net, Xte, Yte, 10); ok(M.flat().reduce((a, b) => a + b, 0) === Xte.length && near(M.reduce((s, row, i) => s + row[i], 0) / Xte.length, ate, 1e-12), 'matrice de confusion cohérente avec la précision');
  const lin = N.create([64, 10], 'relu', 'softmax', 1), ol = N.optimizer(lin, { type: 'adam', lr: 0.01 });
  for (let e = 0; e < 30; e++) N.epoch(lin, ol, Xtr, Ytr, 32, 0, N.rng(4));
  ok(N.accuracy(lin, Xte, Yte) > 0.9, 'chiffres sans couche cachée (régression softmax) : ' + (N.accuracy(lin, Xte, Yte) * 100).toFixed(1) + ' %');
  // gradient vérifié aussi sur ce réseau réel (quelques exemples)
  const small = N.create([64, 12, 10], 'tanh', 'softmax', 3); const gc = N.gradCheck(small, Xtr.slice(0, 5), Ytr.slice(0, 5), 1e-3); ok(gc.worst < 1e-6 && gc.absMax < 1e-8, 'gradient vérifié sur un réseau 64-12-10 (tanh) et de vrais chiffres');
}

console.log(n - ko + ' / ' + n + ' tests réussis');
if (ko) { fails.forEach((f) => console.log('ÉCHEC : ' + f)); process.exit(1); }
