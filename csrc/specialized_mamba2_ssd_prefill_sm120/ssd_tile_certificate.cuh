// Included inside tile_recovery after tiles. All launches use the submitting stream.
#pragma once
using win1306::Aux1734;
using win1306::Out1530;
struct Meta {
  float S, J, ds, dd, sf, df, B, C;
};
struct Seg {
  float a, b, input, jump;
};
struct Vote {
  unsigned top[4];
  int fault, replay;
};  // upper(raw,packed), lower(raw,packed), as nonnegative FP32 bits
__device__ __forceinline__ float wmax(float m) {
  return __uint_as_float(__reduce_max_sync(~0u, __float_as_uint(m)));
}
__device__ __forceinline__ bool src(bf16 z) {
  return win1306::normal_source(__bfloat16_as_ushort(z));
}
__global__ void stats(Args v, Arena* w, Layout l) {
  int s = blockIdx.y, lane = threadIdx.x % 32;
  int64_t t = int64_t(blockIdx.x) * 8 + threadIdx.x / 32;
  Head* hdr = ptr<Head>(w, l, 0);
  if (s >= hdr->count || t >= l.T) return;
  int id = hdr->ids[s], h = id / 2, g = h / (v.nh / 8);
  int64_t u = (t * 8 + g) * 128 + 4 * lane;
  float B = 0, C = 0;
  int bad = 0;
  for (int n = 0; n < 4; n++) {
    bf16 b = v.B[u + n], c = v.C[u + n];
    if (!src(b) || !src(c)) bad |= 1;
    B = fmaxf(B, fabsf(f(b)));
    C = __fadd_ru(C, fabsf(f(c)));
  }
  if (!src(v.x[(t * v.nh + h) * 64 + 32 * (id % 2) + lane])) bad |= 1;
  B = wmax(B);
  for (int d = 1; d < 32; d *= 2) C = __fadd_ru(C, __shfl_down_sync(~0u, C, d));
  bad = __reduce_or_sync(~0u, bad);
  if (!lane) {
    bf16 dt = v.dt[t * v.nh + h];
    if (!src(dt) || !src(v.bias[h]) || !src(v.D[h]) || !win1306::a_source(v.A[h])) bad |= 1;
    double z = __dadd_rn(double(f(dt)), double(f(v.bias[h])));
    const double* q = ptr<double>(w, l, 1) + 2 * (s * l.T + t);
    double a = q[0], d = q[1];
    if (!(z >= -32 && z <= 4096 && a >= 0 && a <= 4096 && d >= 0 && d <= 1)) bad |= 2;
    float sf = __double2float_rn(a), df = __double2float_rn(d);
    ptr<Meta>(w, l, 10)[s * l.T + t] = {
        __double2float_ru(a),
        __double2float_ru(d),
        __double2float_ru(nextafter(fabs(__dsub_rn(a, double(sf))), CUDART_INF)),
        __double2float_ru(nextafter(fabs(__dsub_rn(d, double(df))), CUDART_INF)),
        sf,
        df,
        B,
        C};
    if (bad) atomicOr(&ptr<Vote>(w, l, 15)[s].fault, bad);
  }
}
__device__ __forceinline__ float2 maprow(Meta q, float X, float M, int& bad) {
  Out1530 o;
  Aux1734 u;
  constexpr float uf = 0x1p-24f, ur = 0x1p-53f, G = 1.f + 11.f * 0x1p-23f;
  o.ck(M);
  o.ck(X);
  o.ck(q.B);
  o.ck(q.C);
  o.ck(q.sf);
  o.ck(q.df);
  o.ck(q.S);
  o.ck(q.J);
  o.ck(q.ds);
  o.ck(q.dd);
  float xf = o.V(q.sf, X), xr = o.V(q.S, X), fx = o.rad(xf, uf), rx = o.rad(xr, ur);
  float dx = o.A(o.V(q.ds, X), o.A(fx, rx)), fi = o.V(o.A(xf, fx), q.B), ri = o.V(o.A(xr, rx), q.B);
  float ir = o.rad(ri, ur), ff = o.rad(fi, uf), ei = o.A(o.V(dx, q.B), o.A(ff, ir)),
        of = o.V(M, q.df), off = o.rad(of, uf);
  float p = o.rad(o.A(o.A(of, off), o.A(fi, ff)), uf), k = o.A(ri, ir), mdd = o.V(M, q.dd);
  float a = u.V(G, q.J), b = u.A(ei, mdd);
  b = u.A(b, off);
  b = u.A(b, p);
  b = u.A(b, u.V(ur, k));
  b = u.A(b, u.V(u.V(q.J, M), 0x1.000002p-52f));
  b = u.A(b, 3.f * 0x1p-149f);
  b = u.V(G, u.A(b, 1024.f * 0x1p-149f));
  o.ck(a);
  o.ck(b);
  bad |= u.bad | o.bad;
  return make_float2(a, b);
}
// One warp per (segment,column); adjacent warps own adjacent columns.
template <bool eval>
__global__ void segments(Args v, Arena* w, Layout l) {
  int s = blockIdx.z, c = blockIdx.y * 8 + threadIdx.x / 32, i = threadIdx.x % 32;
  Head* h0 = ptr<Head>(w, l, 0);
  if (s >= h0->count) return;
  int id = h0->ids[s], h = id / 2;
  int64_t seg = blockIdx.x, H = (l.T + 31) / 32, t = 32 * seg + i, ix = (s * l.T + t) * 32 + c,
          si = (s * H + seg) * 32 + c;
  Seg* segs = ptr<Seg>(w, l, 13);
  Out1530 o;
  int bad = 0;
  float2 ab = make_float2(1, 0), mn{};
  Meta q{};
  float X = 0, top[4] = {};
  if (t < l.T) {
    q = ptr<Meta>(w, l, 10)[s * l.T + t];
    mn = ptr<float2>(w, l, 11)[ix];
    X = fabsf(f(v.x[(t * v.nh + h) * 64 + 32 * (id % 2) + c]));
  }
  if (!eval) {
    if (t < l.T) ab = maprow(q, X, mn.x, bad);
    for (int d = 1; d < 32; d *= 2) {
      float a = __shfl_up_sync(~0u, ab.x, d), b = __shfl_up_sync(~0u, ab.y, d);
      if (i >= d && t < l.T) {
        ab.y = o.A(o.V(ab.x, b), ab.y);
        ab.x = o.V(ab.x, a);
      }
    }
    float a = __shfl_up_sync(~0u, ab.x, 1), b = __shfl_up_sync(~0u, ab.y, 1);
    if (t < l.T) {
      ptr<float2>(w, l, 12)[ix] = i ? make_float2(a, b) : make_float2(1, 0);
      if (i == 31 || t + 1 == l.T) {
        segs[si].a = ab.x;
        segs[si].b = ab.y;
      }
    }
  } else {
    if (t < l.T) {
      ab = ptr<float2>(w, l, 12)[ix];
      float E = o.A(o.V(ab.x, segs[si].input), ab.y);
      float R = win1306::radius(q.S, q.J, q.ds, q.dd, q.sf, q.df, X, q.B, q.C, f(v.D[h]), mn.x, E,
                                mn.y, bad);
      ptr<float2>(w, l, 14)[ix] = make_float2(E, R);
      float raw = ptr<float>(w, l, 6)[ix];
      double y = double(f(__float2bfloat16_rn(raw))), lo = __dsub_rd(double(raw), double(R)),
             hi = __dadd_ru(double(raw), double(R));
      o.ck(fabsf(raw));
      o.ck(fabsf(float(y)));
      double pl = fmin(double(f(__double2bfloat16(lo))),
                       double(f(__float2bfloat16_rn(__double2float_rn(lo))))),
             ph = fmax(double(f(__double2bfloat16(hi))),
                       double(f(__float2bfloat16_rn(__double2float_rn(hi)))));
      for (int k = 0; k < 2; k++) {
        double a = k ? pl : lo, b = k ? ph : hi;
        if (!(isfinite(a) && isfinite(b) && a <= b)) bad |= 4;
        top[k] = o.ck(__double2float_ru(fmax(__dsub_ru(y, a), __dsub_ru(b, y))));
        top[2 + k] = o.ck(__double2float_rd(fmax(0., fmax(a, -b))));
      }
    }
    for (int k = 0; k < 4; k++) {
      float m = wmax(fabsf(top[k]));
      if (!i) atomicMax(&ptr<Vote>(w, l, 15)[s].top[k], __float_as_uint(m));
    }
  }
  bad = __reduce_or_sync(~0u, bad | o.bad);
  if (!i && bad) atomicOr(&ptr<Vote>(w, l, 15)[s].fault, bad);
}
__global__ void carries(Arena* w, Layout l) {
  int s = blockIdx.x, c = threadIdx.x;
  if (s >= ptr<Head>(w, l, 0)->count) return;
  int64_t H = (l.T + 31) / 32;
  float E = 0;
  Out1530 o;
  for (int64_t j = 0; j < H; j++) {
    Seg& z = ptr<Seg>(w, l, 13)[(s * H + j) * 32 + c];
    if (!(j % 2)) E = o.A(E, o.ck(z.jump));
    z.input = E;
    E = o.A(o.V(z.a, E), z.b);
  }
  if (o.bad) atomicOr(&ptr<Vote>(w, l, 15)[s].fault, o.bad);
}
__global__ void decide(Arena* w, Layout l) {
  int s = threadIdx.x;
  if (s >= ptr<Head>(w, l, 0)->count) return;
  Vote& v = ptr<Vote>(w, l, 15)[s];
  bool ok = !v.fault;
  for (int k = 0; k < 2; k++) {
    float n = __uint_as_float(v.top[2 + k]);
    ok = ok && n > 0 && __uint_as_float(v.top[k]) <= __fdiv_rd(n, 50.f);
  }
  v.replay = !ok;
}
// Actual original ordered recurrence, same 256-thread geometry as QAOrdered.
__global__ void replay8(Args v, Arena* w, Layout l) {
  Head* h = ptr<Head>(w, l, 0);
  for (int s = 0; s < h->count; s++)
    if (h->ids[s] == int(blockIdx.y) && ptr<Vote>(w, l, 15)[s].replay) {
      extern __shared__ __align__(16) char exact[];
      win1306::pipeline<8>(v.x, v.dt, v.A, v.B, v.C, v.D, v.bias, v.y, v.T, v.nh,
                           int(blockIdx.y) / 2, 32 * (int(blockIdx.y) & 1), exact);
      return;
    }
}
