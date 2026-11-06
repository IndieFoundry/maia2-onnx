import { decodeMoves, decodeValue, type MoveProb } from './decode.js';
import { BOARD_TENSOR_SIZE, encodePosition, eloCategory } from './encode.js';
import { NUM_MOVES } from './vocab.js';

/**
 * Structural subset of onnxruntime's Tensor / InferenceSession, so the same code
 * works with `onnxruntime-node`, `onnxruntime-web` and `onnxruntime-react-native`
 * without importing any of them.
 */
export interface TensorLike {
  readonly data: unknown;
  readonly dims: readonly number[];
}
export interface TensorConstructor {
  new (type: 'float32', data: Float32Array, dims: readonly number[]): TensorLike;
  new (type: 'int64', data: BigInt64Array, dims: readonly number[]): TensorLike;
}
export interface SessionLike {
  run(feeds: Record<string, TensorLike>): Promise<Record<string, TensorLike>>;
}

export interface MaiaOptions {
  /** An InferenceSession created on maia2-*.onnx (fp32 or int8). */
  session: SessionLike;
  /** The `Tensor` class of the same onnxruntime package (`ort.Tensor`). */
  Tensor: TensorConstructor;
}

export interface Prediction {
  /** All legal moves, sorted by decreasing probability (sum = 1). */
  moves: MoveProb[];
  /** Expected score for WHITE, ~P(win) + 0.5*P(draw) (maia2's `win_prob`, same convention as inference_each). */
  winProb: number;
  /** Expected score for the side to move. */
  sideToMoveWinProb: number;
}

export interface PredictInput {
  fen: string;
  /** Elo of the side to move. */
  eloSelf: number;
  /** Elo of the opponent. */
  eloOppo: number;
}

export interface Maia {
  predict(fen: string, eloSelf: number, eloOppo: number): Promise<Prediction>;
  /** One session.run for all inputs (the model has a dynamic batch axis). */
  predictBatch(inputs: readonly PredictInput[]): Promise<Prediction[]>;
}

export function createMaia({ session, Tensor }: MaiaOptions): Maia {
  async function predictBatch(inputs: readonly PredictInput[]): Promise<Prediction[]> {
    const n = inputs.length;
    if (n === 0) return [];
    const boards = new Float32Array(n * BOARD_TENSOR_SIZE);
    const eloSelf = new BigInt64Array(n);
    const eloOppo = new BigInt64Array(n);
    inputs.forEach((inp, i) => {
      boards.set(encodePosition(inp.fen).planes, i * BOARD_TENSOR_SIZE);
      eloSelf[i] = BigInt(eloCategory(inp.eloSelf));
      eloOppo[i] = BigInt(eloCategory(inp.eloOppo));
    });
    const out = await session.run({
      boards: new Tensor('float32', boards, [n, 18, 8, 8]),
      elo_self: new Tensor('int64', eloSelf, [n]),
      elo_oppo: new Tensor('int64', eloOppo, [n]),
    });
    const logits = out['logits_move']?.data as Float32Array | undefined;
    const values = out['logits_value']?.data as Float32Array | undefined;
    if (!logits || !values || logits.length !== n * NUM_MOVES || values.length !== n) {
      throw new Error('Unexpected model outputs (expected logits_move [n,1880] and logits_value [n])');
    }
    return inputs.map((inp, i) => ({
      moves: decodeMoves(logits.subarray(i * NUM_MOVES, (i + 1) * NUM_MOVES), inp.fen),
      ...decodeValue(values[i]!, inp.fen),
    }));
  }

  return {
    predictBatch,
    async predict(fen, eloSelf, eloOppo) {
      const [p] = await predictBatch([{ fen, eloSelf, eloOppo }]);
      return p!;
    },
  };
}
