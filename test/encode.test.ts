import { describe, expect, it } from 'vitest';
import { cleanCastlingFen, encodePosition, eloCategory, legalMoves, mirrorUci, ALL_MOVES } from '../src/index.js';
import { readJson } from './helpers.js';

interface EncodingFixture {
  positions: { fen: string; shape: number[]; ones: number[]; legal: number[] }[];
  eloCategories: [number, number][];
}
const fx = readJson<EncodingFixture>('fixtures/encoding.json');

describe('encodePosition matches maia2.inference.preprocessing()', () => {
  it.each(fx.positions.map((p) => [p.fen, p] as const))('%s', (_fen, p) => {
    expect(p.shape).toEqual([18, 8, 8]);
    const { planes } = encodePosition(p.fen);
    const ones: number[] = [];
    planes.forEach((v, i) => {
      if (v !== 0) {
        expect(v).toBe(1);
        ones.push(i);
      }
    });
    expect(ones).toEqual(p.ones);
  });
});

describe('legal move indices match maia2 legal-move mask', () => {
  it.each(fx.positions.map((p) => [p.fen, p] as const))('%s', (_fen, p) => {
    const idx = legalMoves(p.fen).map((m) => m.index).sort((a, b) => a - b);
    expect(idx).toEqual(p.legal);
  });
});

describe('eloCategory matches maia2.utils.map_to_category', () => {
  it.each(fx.eloCategories)('%d -> %d', (elo, cat) => {
    expect(eloCategory(elo)).toBe(cat);
  });
  it('rejects NaN', () => expect(() => eloCategory(Number.NaN)).toThrow());
});

describe('vocabulary', () => {
  it('has 1880 unique moves, promotions only from rank 7', () => {
    expect(ALL_MOVES.length).toBe(1880);
    expect(new Set(ALL_MOVES).size).toBe(1880);
    expect(ALL_MOVES.filter((m) => m.length === 5).every((m) => m[1] === '7' && m[3] === '8')).toBe(true);
  });
  it('mirrorUci', () => {
    expect(mirrorUci('e2e1q')).toBe('e7e8q');
    expect(mirrorUci('e8g8')).toBe('e1g1');
  });
});

describe('cleanCastlingFen', () => {
  it('drops flags whose rook or king moved', () => {
    expect(cleanCastlingFen('r3k3/8/8/8/8/8/8/4K2R w KQkq - 0 1')).toBe('r3k3/8/8/8/8/8/8/4K2R w Kq - 0 1');
    expect(cleanCastlingFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')).toContain(' KQkq ');
    expect(cleanCastlingFen('8/8/8/8/8/8/8/R3K3 w K - 0 1')).toContain(' - ');
  });
});
