/* Réseau de neurones écrit à la main : moteur, sans dépendance.
   Perceptron multicouche (couches denses), activations ReLU / Leaky ReLU / tanh / sigmoïde / linéaire,
   sorties sigmoïde + entropie croisée binaire, softmax + entropie croisée, ou linéaire + erreur quadratique.
   Rétropropagation du gradient écrite à la main et vérifiée par différences finies.
   Optimiseurs : descente de gradient (mini-lots), moment, RMSprop, Adam ; régularisation L2.
   Jeux de données 2D générés (cercles, XOR, lunes, spirales, gaussiennes, damier) et chiffres manuscrits 8×8 (UCI).
   Utilisable dans le navigateur (window.NN) et avec Node (module.exports). */
(function (root, factory) {
  const M = factory();
  if (typeof module === 'object' && module.exports) module.exports = M;
  else root.NN = M;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  // ---------------------------------------------------------------- hasard reproductible
  function rng(seed) { let a = seed >>> 0 || 1; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = r(); if (u < 1e-300) u = 1e-300; return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  function shuffle(arr, r) { for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = arr[i]; arr[i] = arr[j]; arr[j] = t; } return arr; }

  // ---------------------------------------------------------------- activations : f(z) et f'(z) (exprimée avec z et a = f(z))
  const ACT = {
    relu: { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
    leaky: { f: (z) => (z > 0 ? z : 0.01 * z), d: (z) => (z > 0 ? 1 : 0.01) },
    tanh: { f: (z) => Math.tanh(z), d: (z, a) => 1 - a * a },
    sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), d: (z, a) => a * (1 - a) },
    linear: { f: (z) => z, d: () => 1 },
  };

  // ---------------------------------------------------------------- réseau
  // sizes = [entrées, cachées…, sorties] ; act = activation des couches cachées ; out = 'sigmoid' | 'softmax' | 'linear'
  // W[l] : matrice n[l+1] × n[l] (ligne = neurone d'arrivée), b[l] : n[l+1]
  function create(sizes, act, out, seed, init) {
    const r = rng(seed || 1), W = [], b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nin = sizes[l], nout = sizes[l + 1], last = l === sizes.length - 2;
      const he = !last && (act === 'relu' || act === 'leaky') && init !== 'xavier';
      const sd = he ? Math.sqrt(2 / nin) : Math.sqrt(2 / (nin + nout)); // He (ReLU) ou Glorot/Xavier (tanh, sigmoïde, sortie)
      W.push(Float64Array.from({ length: nout * nin }, () => gauss(r) * sd));
      b.push(new Float64Array(nout));
    }
    return { sizes: sizes.slice(), act, out, W, b };
  }
  const nParams = (net) => net.W.reduce((s, w, l) => s + w.length + net.b[l].length, 0);
  function clone(net) { return { sizes: net.sizes.slice(), act: net.act, out: net.out, W: net.W.map((w) => Float64Array.from(w)), b: net.b.map((v) => Float64Array.from(v)) }; }
  function softmax(z) { let m = -Infinity; for (const v of z) m = Math.max(m, v); let s = 0; const a = new Float64Array(z.length); for (let i = 0; i < z.length; i++) { a[i] = Math.exp(z[i] - m); s += a[i]; } for (let i = 0; i < z.length; i++) a[i] /= s; return a; }
  // propagation avant d'un exemple : renvoie les pré-activations z[l] et les activations a[l] (a[0] = x)
  function forward(net, x) {
    const L = net.W.length, a = [Float64Array.from(x)], z = [null];
    for (let l = 0; l < L; l++) {
      const nin = net.sizes[l], nout = net.sizes[l + 1], W = net.W[l], prev = a[l], zl = new Float64Array(nout);
      for (let j = 0; j < nout; j++) { let s = net.b[l][j]; const o = j * nin; for (let i = 0; i < nin; i++) s += W[o + i] * prev[i]; zl[j] = s; }
      let al;
      if (l === L - 1) al = net.out === 'softmax' ? softmax(zl) : net.out === 'sigmoid' ? zl.map(ACT.sigmoid.f) : Float64Array.from(zl);
      else { const f = ACT[net.act].f; al = zl.map(f); }
      z.push(zl); a.push(al);
    }
    return { z, a };
  }
  const predict = (net, x) => { const c = forward(net, x); return c.a[c.a.length - 1]; };
  // perte d'un exemple : y = indice de classe (sigmoid/softmax) ou valeur(s) cible (linear)
  function lossOf(net, out, y) {
    const eps = 1e-12;
    if (net.out === 'sigmoid') { const p = Math.min(1 - eps, Math.max(eps, out[0])); return -(y * Math.log(p) + (1 - y) * Math.log(1 - p)); }
    if (net.out === 'softmax') return -Math.log(Math.max(eps, out[y]));
    let s = 0; const t = typeof y === 'number' ? [y] : y; for (let i = 0; i < out.length; i++) s += 0.5 * (out[i] - t[i]) ** 2; return s;
  }
  const l2Of = (net) => net.W.reduce((s, w) => s + w.reduce((u, v) => u + v * v, 0), 0) / 2;
  // rétropropagation d'un exemple : ajoute ∂L/∂W et ∂L/∂b dans g (pour chaque couche)
  // couche de sortie : avec ces couples (sigmoïde + entropie croisée binaire, softmax + entropie croisée, linéaire + quadratique), δ = a − y
  function backward(net, cache, y, g) {
    const L = net.W.length, { z, a } = cache;
    let delta = Float64Array.from(a[L]);
    if (net.out === 'softmax') delta[y] -= 1;
    else if (net.out === 'sigmoid') delta[0] -= y;
    else { const t = typeof y === 'number' ? [y] : y; for (let i = 0; i < delta.length; i++) delta[i] -= t[i]; }
    for (let l = L - 1; l >= 0; l--) {
      const nin = net.sizes[l], nout = net.sizes[l + 1], W = net.W[l], prev = a[l], gW = g.W[l], gb = g.b[l];
      for (let j = 0; j < nout; j++) { const d = delta[j], o = j * nin; gb[j] += d; for (let i = 0; i < nin; i++) gW[o + i] += d * prev[i]; }
      if (l > 0) {
        const nd = new Float64Array(nin), df = ACT[net.act].d;
        for (let i = 0; i < nin; i++) { let s = 0; for (let j = 0; j < nout; j++) s += W[j * nin + i] * delta[j]; nd[i] = s * df(z[l][i], a[l][i]); }
        delta = nd;
      }
    }
    return delta;
  }
  const zeroGrad = (net) => ({ W: net.W.map((w) => new Float64Array(w.length)), b: net.b.map((v) => new Float64Array(v.length)) });
  // perte moyenne + gradient moyen sur un lot (avec L2 : + λ/2·ΣW², biais non régularisés)
  function batchGrad(net, X, Y, idx, lambda) {
    const g = zeroGrad(net); let loss = 0;
    const ids = idx || X.map((_, i) => i);
    for (const i of ids) { const c = forward(net, X[i]); loss += lossOf(net, c.a[c.a.length - 1], Y[i]); backward(net, c, Y[i], g); }
    const n = ids.length;
    g.W.forEach((w, l) => { for (let k = 0; k < w.length; k++) w[k] = w[k] / n + (lambda || 0) * net.W[l][k]; });
    g.b.forEach((v) => { for (let k = 0; k < v.length; k++) v[k] /= n; });
    return { loss: loss / n + (lambda || 0) * l2Of(net), g };
  }
  function meanLoss(net, X, Y, lambda) { let s = 0; for (let i = 0; i < X.length; i++) s += lossOf(net, predict(net, X[i]), Y[i]); return s / X.length + (lambda || 0) * l2Of(net); }
  // vérification du gradient par différences centrées : erreur relative maximale |g − ĝ| / max(|g| + |ĝ|, petit)
  function gradCheck(net, X, Y, lambda, h) {
    const eps = h || 1e-5, { g } = batchGrad(net, X, Y, null, lambda);
    // erreur relative sur les gradients non négligeables (> 10⁻⁴ × le plus grand), erreur absolue sur tous
    let gmax = 0; g.W.concat(g.b).forEach((arr) => arr.forEach((v) => (gmax = Math.max(gmax, Math.abs(v)))));
    let worst = 0, absMax = 0, count = 0;
    const check = (arr, garr) => { for (let k = 0; k < arr.length; k++) { const v = arr[k]; arr[k] = v + eps; const lp = meanLoss(net, X, Y, lambda); arr[k] = v - eps; const lm = meanLoss(net, X, Y, lambda); arr[k] = v; const num = (lp - lm) / (2 * eps), an = garr[k], d = Math.abs(num - an); absMax = Math.max(absMax, d); if (Math.max(Math.abs(num), Math.abs(an)) > 1e-4 * gmax) worst = Math.max(worst, d / (Math.abs(num) + Math.abs(an))); count++; } };
    net.W.forEach((w, l) => check(w, g.W[l])); net.b.forEach((v, l) => check(v, g.b[l]));
    return { worst, absMax, count, gmax };
  }

  // plus petite |pré-activation| des couches cachées : ReLU n'est pas dérivable en 0, la vérification par différences finies
  // n'a de sens que loin de ce point anguleux
  function minPreact(net, X) { let m = Infinity; for (const x of X) { const c = forward(net, x); for (let l = 1; l < c.z.length - 1; l++) for (const v of c.z[l]) m = Math.min(m, Math.abs(v)); } return m; }

  // ---------------------------------------------------------------- optimiseurs
  // o : { type: 'sgd' | 'momentum' | 'rmsprop' | 'adam', lr, beta (moment), beta1, beta2, eps }
  function optimizer(net, o) {
    const opt = Object.assign({ type: 'adam', lr: 0.01, beta: 0.9, beta1: 0.9, beta2: 0.999, rho: 0.9, eps: 1e-8 }, o || {});
    const m = zeroGrad(net), v = zeroGrad(net); let t = 0;
    opt.step = (g) => {
      t++;
      const upd = (P, G, M, V) => {
        for (let k = 0; k < P.length; k++) {
          const gk = G[k];
          if (opt.type === 'sgd') P[k] -= opt.lr * gk;
          else if (opt.type === 'momentum') { M[k] = opt.beta * M[k] + gk; P[k] -= opt.lr * M[k]; }
          else if (opt.type === 'rmsprop') { V[k] = opt.rho * V[k] + (1 - opt.rho) * gk * gk; P[k] -= (opt.lr * gk) / (Math.sqrt(V[k]) + opt.eps); }
          else { M[k] = opt.beta1 * M[k] + (1 - opt.beta1) * gk; V[k] = opt.beta2 * V[k] + (1 - opt.beta2) * gk * gk; const mh = M[k] / (1 - opt.beta1 ** t), vh = V[k] / (1 - opt.beta2 ** t); P[k] -= (opt.lr * mh) / (Math.sqrt(vh) + opt.eps); }
        }
      };
      net.W.forEach((w, l) => upd(w, g.W[l], m.W[l], v.W[l])); net.b.forEach((b, l) => upd(b, g.b[l], m.b[l], v.b[l]));
    };
    opt.t = () => t;
    return opt;
  }
  // une époque : mini-lots dans un ordre mélangé ; renvoie la perte moyenne des lots
  function epoch(net, opt, X, Y, batch, lambda, r) {
    const idx = shuffle(X.map((_, i) => i), r || Math.random), bs = Math.max(1, Math.min(batch || 16, X.length));
    let s = 0, n = 0;
    for (let k = 0; k < idx.length; k += bs) { const { loss, g } = batchGrad(net, X, Y, idx.slice(k, k + bs), lambda); opt.step(g); s += loss; n++; }
    return s / n;
  }
  function accuracy(net, X, Y) {
    let ok = 0;
    for (let i = 0; i < X.length; i++) { const p = predict(net, X[i]); const c = net.out === 'sigmoid' ? (p[0] >= 0.5 ? 1 : 0) : argmax(p); if (c === Y[i]) ok++; }
    return X.length ? ok / X.length : 0;
  }
  function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }
  function confusion(net, X, Y, K) {
    const M = Array.from({ length: K }, () => new Array(K).fill(0));
    for (let i = 0; i < X.length; i++) { const p = predict(net, X[i]), c = net.out === 'sigmoid' ? (p[0] >= 0.5 ? 1 : 0) : argmax(p); M[Y[i]][c]++; }
    return M;
  }

  // ---------------------------------------------------------------- jeux de données 2D (points dans [−1, 1]²)
  const DATASETS = ['circle', 'xor', 'moons', 'spiral', 'spiral3', 'gauss', 'checker'];
  function dataset(kind, n, noise, seed) {
    const r = rng(seed || 3), P = [], Y = [], nz = noise || 0, U = (a, b) => a + (b - a) * r();
    for (let i = 0; i < n; i++) {
      let x, y, c;
      if (kind === 'circle') { c = i % 2; const rad = c ? U(0, 0.45) : U(0.65, 0.95), t = U(0, 2 * Math.PI); x = rad * Math.cos(t); y = rad * Math.sin(t); }
      else if (kind === 'xor') { x = U(-0.95, 0.95); y = U(-0.95, 0.95); if (Math.abs(x) < 0.05) x += Math.sign(x || 1) * 0.05; if (Math.abs(y) < 0.05) y += Math.sign(y || 1) * 0.05; c = x * y > 0 ? 0 : 1; }
      else if (kind === 'moons') { c = i % 2; const t = U(0, Math.PI); if (c === 0) { x = Math.cos(t) - 0.5; y = Math.sin(t) - 0.25; } else { x = 0.5 - Math.cos(t); y = 0.25 - Math.sin(t); } x *= 0.62; y *= 0.62; }
      else if (kind === 'spiral' || kind === 'spiral3') { const K = kind === 'spiral' ? 2 : 3; c = i % K; const t = U(0, 1), rad = 0.08 + 0.85 * t, ang = 1.75 * t * 2 * Math.PI + (c * 2 * Math.PI) / K; x = rad * Math.cos(ang); y = rad * Math.sin(ang); }
      else if (kind === 'gauss') { c = i % 2; x = (c ? 0.4 : -0.4) + 0.22 * gauss(r); y = (c ? 0.4 : -0.4) + 0.22 * gauss(r); }
      else { x = U(-0.95, 0.95); y = U(-0.95, 0.95); c = (Math.floor((x + 1) * 2) + Math.floor((y + 1) * 2)) % 2; }
      x += nz * 0.12 * gauss(r); y += nz * 0.12 * gauss(r);
      P.push([x, y]); Y.push(c);
    }
    return { P, Y, K: kind === 'spiral3' ? 3 : 2 };
  }
  // découpage apprentissage / test reproductible
  function split(n, testFrac, seed) { const idx = shuffle(Array.from({ length: n }, (_, i) => i), rng(seed || 11)), nt = Math.round(n * testFrac); return { test: idx.slice(0, nt), train: idx.slice(nt) }; }
  // caractéristiques d'entrée calculées à partir de (x, y)
  const FEATURES = { x: (p) => p[0], y: (p) => p[1], x2: (p) => p[0] * p[0], y2: (p) => p[1] * p[1], xy: (p) => p[0] * p[1], sinx: (p) => Math.sin(Math.PI * p[0]), siny: (p) => Math.sin(Math.PI * p[1]) };
  const featurize = (p, feats) => feats.map((f) => FEATURES[f](p));

  // ---------------------------------------------------------------- chiffres manuscrits 8×8
  function parseDigits(str) {
    const al = '0123456789abcdefg', X = [], Y = [];
    for (let k = 0; k + 65 <= str.length; k += 65) { const x = new Float64Array(64); for (let i = 0; i < 64; i++) x[i] = al.indexOf(str[k + i]) / 16; X.push(x); Y.push(+str[k + 64]); }
    return { X, Y };
  }
  // même principe que le prétraitement UCI : image binaire 32×32 → nombre de pixels allumés par bloc 4×4 (0…16)
  function blocks32to8(bits) { const x = new Float64Array(64); for (let by = 0; by < 8; by++) for (let bx = 0; bx < 8; bx++) { let s = 0; for (let y = 0; y < 4; y++) for (let xx = 0; xx < 4; xx++) s += bits[(by * 4 + y) * 32 + bx * 4 + xx] ? 1 : 0; x[by * 8 + bx] = s / 16; } return x; }

  return { rng, gauss, shuffle, ACT, create, clone, nParams, softmax, forward, predict, lossOf, l2Of, backward, zeroGrad, batchGrad, meanLoss, gradCheck, minPreact, optimizer, epoch, accuracy, argmax, confusion, DATASETS, dataset, split, FEATURES, featurize, parseDigits, blocks32to8 };
});
