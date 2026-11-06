"""Generate golden fixtures from the PyTorch Maia-2 model using maia2's own helpers.

Usage (from the repo root):
    .venv/bin/python python/make_fixtures.py [rapid]

Writes:
    fixtures/positions.json         the 60 test positions (category, fen)
    fixtures/encoding.json          maia2.inference.preprocessing() output per position
                                    (indices of the 1s in the 18x8x8 tensor, legal move
                                    indices, elo categories for a few Elo values)
    fixtures/golden-<type>.json     inference_each() output (rounded, as maia2 returns it)
                                    plus unrounded top-10 legal-move probabilities and the
                                    raw value logit, for each position x Elo pair
"""

import json
import sys
from pathlib import Path

import chess
import torch
from maia2 import inference
from maia2 import model as maia_model
from maia2.utils import map_to_category, mirror_move

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"
WEIGHTS = ROOT / "maia2_models"

ELO_PAIRS = [
    (1100, 1100),
    (1500, 1500),
    (1900, 1900),
    (1100, 1900),
    (1900, 1100),
    (800, 2400),
]
ELO_CATEGORY_PROBES = [0, 600, 1099, 1100, 1150, 1199, 1200, 1499.5, 1500, 1899, 1999, 1999.9, 2000, 2500, 3200]


def from_moves(san_moves: str) -> str:
    b = chess.Board()
    for san in san_moves.split():
        b.push_san(san)
    return b.fen()


OPENINGS = [
    chess.STARTING_FEN,
    from_moves("e4"),
    from_moves("d4"),
    from_moves("e4 e5"),
    from_moves("e4 c5"),
    from_moves("e4 e5 Nf3 Nc6"),
    from_moves("e4 e5 Nf3 Nc6 Bb5"),
    from_moves("e4 e5 Nf3 Nc6 Bc4 Bc5"),
    from_moves("d4 d5 c4"),
    from_moves("d4 Nf6 c4 e6 Nc3 Bb4"),
    from_moves("e4 c6 d4 d5"),
    from_moves("e4 e6 d4 d5 Nc3"),
    from_moves("e4 d5 exd5 Qxd5 Nc3"),
    from_moves("c4 e5 Nc3 Nf6 g3"),
    from_moves("Nf3 d5 g3 Nf6 Bg2 c6 O-O"),
    from_moves("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"),
    from_moves("e4 e5 d4 exd4 c3"),
    from_moves("e4 e5 f4"),
    from_moves("e4 d6 d4 Nf6 Nc3 g6 f4"),
    from_moves("d4 f5 e4 fxe4 Nc3"),
]

MIDDLEGAMES = [
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2QKB1R w KQ - 0 9",
    "r2q1rk1/pp1bbppp/2n1pn2/3p4/3P4/2PBPN2/PP1N1PPP/R2Q1RK1 b - - 3 10",
    "r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/5N1P/PPBN1PP1/R1BQR1K1 w - - 0 13",
    "2rq1rk1/pp1bppbp/3p1np1/4n3/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 5 12",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bqr1k1/ppp2ppp/2np1n2/2b1p3/2B1P3/2PP1N2/PP1N1PPP/R1BQR1K1 b - - 3 8",
    "r2qkb1r/pp2pppp/2n2n2/3p1b2/3P1B2/2N1PN2/PP3PPP/R2QKB1R b KQkq - 2 7",
    "rnbqkb1r/pp3ppp/4pn2/2pp4/3P4/2PBPN2/PP3PPP/RNBQK2R b KQkq - 1 5",
    "r3k2r/ppp2ppp/2n1bn2/2bqp3/8/2NP1N2/PPP1BPPP/R1BQK2R w KQkq - 4 8",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
    "2r2rk1/1bq1bppp/p2ppn2/1p6/3NP3/1BN1B3/PPP1QPPP/2KR3R w - - 4 14",
    "r1bq1rk1/pppn1pbp/3p1np1/4p3/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8",
    "r4rk1/pp3ppp/2p1b3/4P3/2P1n3/4B3/PP4PP/R3KB1R w KQ - 0 16",
    "3r2k1/pp3ppp/2p5/8/2P1n3/1P2B3/P5PP/3R2K1 b - - 1 24",
    "r1b1k2r/ppppqppp/2n2n2/2b5/2B1P3/2N2N2/PPP2PPP/R1BQK2R w KQkq - 6 6",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7",
    "1r3rk1/5ppp/p1q1p3/2p1Pb2/2Pp4/3P2P1/PQ3PBP/R4RK1 b - - 2 21",
    "4rrk1/pp4pp/2p1q3/3pNp2/3P1Q2/2P3P1/PP3P1P/4RRK1 w - - 0 22",
    "r3r1k1/pp1q1ppp/2p2n2/3p4/3P4/2NQPB2/PP3PPP/R4RK1 b - - 5 15",
]

ENDGAMES = [
    "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",  # KP vs K
    "8/8/8/4k3/8/8/4P3/4K3 b - - 0 1",  # KP vs K, black to move
    "8/4P3/8/8/2k5/8/8/4K3 w - - 0 1",  # promotion for White
    "4k3/8/8/8/8/2K5/4p3/8 b - - 0 1",  # promotion for Black (mirroring of promotions)
    "8/8/8/8/8/5k2/8/R3K3 w - - 0 1",  # KR vs K
    "8/8/8/8/8/5K2/8/r3k3 b - - 0 1",  # KR vs K, black to move
    "8/8/8/8/3k4/8/8/4K2Q w - - 0 1",  # KQ vs K
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",  # back-rank
    "8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 1",  # pawn endgame
    "8/8/1p6/1P6/8/2k5/8/2K5 b - - 0 1",  # opposition
    "R7/P4k2/8/8/8/8/r5PK/8 w - - 0 1",  # Vancura-type rook ending
    "1K6/1P1k4/8/8/8/8/r7/2R5 w - - 0 1",  # Lucena
    "8/8/8/8/4k3/8/r7/4K2R w K - 0 1",  # castling right in endgame
    "8/8/4k3/8/8/2B5/4K3/7N w - - 0 1",  # KBN vs K
    "8/8/8/3k4/8/8/2N5/1BK5 b - - 0 1",  # KBN vs K, black to move
    "6k1/8/6K1/6P1/8/8/8/8 w - - 0 1",  # KP vs K, key squares
    "8/2p5/8/8/8/5K2/3k4/8 b - - 0 1",  # runaway pawn black
    "8/8/8/8/1q6/8/5k2/7K b - - 0 1",  # KQ vs K near stalemate
    "3q4/8/8/8/8/8/2Q5/k1K5 w - - 0 1",  # QvQ
    "8/8/3k4/3p4/3P4/3K4/8/8 w - - 0 1",  # blocked pawns, opposition
]

# Encoding-only edge cases (not in the golden set): castling flags that python-chess
# drops because king/rook left their squares, Black castling rights + ep while Black moves.
ENCODING_EXTRA = [
    "r3k3/8/8/8/8/8/8/4K2R w KQkq - 0 1",
    "1r2k2r/8/8/8/8/8/8/R3K1R1 b KQkq - 0 1",
    "r3k2r/8/8/8/3pP3/8/8/R3K2R b KQkq e3 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
]

POSITIONS = (
    [("opening", f) for f in OPENINGS]
    + [("middlegame", f) for f in MIDDLEGAMES]
    + [("endgame", f) for f in ENDGAMES]
)


def check_positions():
    for cat, fen in POSITIONS:
        b = chess.Board(fen)
        assert b.is_valid(), (cat, fen, b.status())
        assert any(b.legal_moves), (cat, fen)


def ones(t: torch.Tensor):
    return torch.nonzero(t.flatten()).flatten().tolist()


def main(kind: str):
    check_positions()
    FIX.mkdir(exist_ok=True)
    prepared = inference.prepare()
    all_moves_dict, elo_dict, all_moves_rev = prepared

    (FIX / "positions.json").write_text(
        json.dumps([{"category": c, "fen": f} for c, f in POSITIONS], indent=1)
    )

    enc = []
    for fen in [f for _, f in POSITIONS] + ENCODING_EXTRA:
        board_input, _, _, legal = inference.preprocessing(fen, 1500, 1500, elo_dict, all_moves_dict)
        enc.append(
            {
                "fen": fen,
                "shape": list(board_input.shape),
                "ones": ones(board_input),
                "legal": ones(legal),
            }
        )
    (FIX / "encoding.json").write_text(
        json.dumps(
            {
                "positions": enc,
                "eloCategories": [[e, map_to_category(e, elo_dict)] for e in ELO_CATEGORY_PROBES],
            }
        )
    )

    m = maia_model.from_pretrained(kind, "cpu", save_root=str(WEIGHTS))
    m.eval()
    golden = []
    for cat, fen in POSITIONS:
        for es, eo in ELO_PAIRS:
            move_probs, win_prob = inference.inference_each(m, prepared, fen, es, eo)
            # Same computation as inference_each, kept unrounded.
            board_input, cs, co, legal = inference.preprocessing(fen, es, eo, elo_dict, all_moves_dict)
            with torch.no_grad():
                lm, _, lv = m(board_input.unsqueeze(0), torch.tensor([cs]), torch.tensor([co]))
            probs = inference._masked_softmax(lm, legal.unsqueeze(0))[0]
            black = fen.split(" ")[1] == "b"
            idx = legal.nonzero().flatten().tolist()
            ranked = sorted(
                ((mirror_move(all_moves_rev[i]) if black else all_moves_rev[i], probs[i].item()) for i in idx),
                key=lambda kv: kv[1],
                reverse=True,
            )
            golden.append(
                {
                    "category": cat,
                    "fen": fen,
                    "eloSelf": es,
                    "eloOppo": eo,
                    "top10": [{"uci": u, "p": p} for u, p in ranked[:10]],
                    "numLegal": len(idx),
                    "valueLogit": lv.item(),
                    "maia2": {"top10": list(move_probs.items())[:10], "winProb": win_prob},
                }
            )
    (FIX / f"golden-{kind}.json").write_text(json.dumps(golden))
    print(f"{len(POSITIONS)} positions, {len(golden)} golden predictions for {kind}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rapid")
