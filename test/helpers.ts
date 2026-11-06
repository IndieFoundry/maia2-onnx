import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export const root = fileURLToPath(new URL('..', import.meta.url));
export const readJson = <T>(rel: string): T => JSON.parse(readFileSync(root + rel, 'utf8')) as T;
export const exists = (rel: string) => existsSync(root + rel);

export interface Golden {
  category: 'opening' | 'middlegame' | 'endgame';
  fen: string;
  eloSelf: number;
  eloOppo: number;
  top10: { uci: string; p: number }[];
  numLegal: number;
  valueLogit: number;
  maia2: { top10: [string, number][]; winProb: number };
}
