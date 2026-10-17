"""New bounded shared residual adapter; explicitly not recovered Stage10 code."""
import copy
import math
import random

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from torch import nn
from torch.nn import functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BoundedAdapter(nn.Module):
    def __init__(self, dimension, rank, radius):
        super().__init__()
        self.radius = radius
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        value = F.normalize(value.float(), dim=-1)
        correction = self.up(torch.tanh(self.down(value)))
        correction = correction / correction.norm(dim=-1, keepdim=True).clamp_min(1.0)
        return F.normalize(value + self.radius * correction, dim=-1)


def scores(cache, rows, head=None, device='cpu', scale=2.5, clamp=True):
    images, captions = [], []
    for row in rows:
        images.append(cache['image'][cache['image_index'][row['image']]])
        captions.append(cache['text'][cache['text_index'][row['caption']]])
    result = []
    for start in range(0, len(rows), 512):
        image = torch.as_tensor(np.asarray(images[start:start + 512]), device=device).float()
        text = torch.as_tensor(np.asarray(captions[start:start + 512]), device=device).float()
        if head is not None:
            with torch.inference_mode():
                image, text = head(image), head(text)
        value = (F.normalize(image, dim=-1) * F.normalize(text, dim=-1)).sum(-1)
        if clamp:
            value = value.clamp_min(0)
        result.extend((value * scale).detach().cpu().tolist())
    return np.asarray(result, dtype=np.float64)


def prediction_pack(cache, records, head=None, device='cpu', scale=2.5, clamp=True):
    candidate = scores(cache, records, head, device, scale, clamp)
    negatives = [dict(r, caption=r['negative_caption']) for r in records if r['domain'] == 'sugar']
    negative = scores(cache, negatives, head, device, scale, clamp) if negatives else np.array([])
    return candidate, negative


def correlation(value, rating):
    statistic = float(kendalltau(value, rating, variant='c').statistic)
    return statistic if math.isfinite(statistic) else None


def metric(rows, prediction, negatives=None):
    if not rows:
        raise ValueError('Empty evaluation partition')
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions')
    if rows[0]['domain'] == 'sugar':
        wins = prediction > negatives
        cats = {c: float(np.mean([w for r, w in zip(rows, wins) if r['category'] == c]))
                for c in sorted({r['category'] for r in rows})}
        return {'accuracy': float(wins.mean()), 'macro_accuracy': float(np.mean(list(cats.values()))),
                'categories': cats, 'n_pairs': len(rows), 'ties': int(np.sum(prediction == negatives))}
    raw_s, raw_y, means = [], [], []
    groups = {}
    for i, (row, score) in enumerate(zip(rows, prediction)):
        raw_s.extend([score] * len(row['ratings']))
        raw_y.extend(row['ratings'])
        means.append(float(np.mean(row['ratings'])))
        groups.setdefault(row['image'], []).append(i)
    wins = []
    for indices in groups.values():
        for pos, a in enumerate(indices):
            for b in indices[pos + 1:]:
                if means[a] == means[b]:
                    continue
                delta = (prediction[a] - prediction[b]) * (means[a] - means[b])
                wins.append(1.0 if delta > 0 else 0.5 if delta == 0 else 0.0)
    rho = float(spearmanr(prediction, means).statistic)
    return {'raw_vote_tau_c': correlation(raw_s, raw_y), 'caption_mean_tau_c': correlation(prediction, means),
            'caption_mean_spearman': rho if math.isfinite(rho) else None,
            'caption_ranking_accuracy': float(np.mean(wins)) if wins else None,
            'ranking_pairs': len(wins), 'n_candidates': len(rows), 'n_votes': len(raw_y)}


def partition_metrics(records, prediction, negative):
    result = {}
    for domain in ['sugar', 'flickr', 'polaris']:
        indices = [i for i, r in enumerate(records) if r['domain'] == domain]
        result[domain] = metric([records[i] for i in indices], prediction[indices], negative if domain == 'sugar' else None)
    return result


def validation_vector(metrics):
    values = [metrics['sugar']['accuracy'], metrics['flickr']['caption_mean_tau_c'], metrics['polaris']['caption_mean_tau_c']]
    if any(v is None for v in values):
        raise ValueError('Undefined validation correlation: do not select on a constant prediction/label')
    return np.asarray(values)


def pair_table(rows, domain, cfg, seed, phase):
    """Only caller-supplied train/development rows can enter this table."""
    if any(r['split'] == 'test' for r in rows):
        raise ValueError('Test labels cannot enter fitting')
    if domain == 'sugar':
        return [(r['image'], r['caption'], r['negative_caption'], 1.0) for r in rows if r['domain'] == domain]
    groups = {}
    for row in rows:
        if row['domain'] == domain:
            groups.setdefault(row['image'], []).append(row)
    pairs = []
    for name, candidates in sorted(groups.items()):
        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                ma, mb = np.mean(a['ratings']), np.mean(b['ratings'])
                if ma == mb:
                    continue
                better, worse = (a, b) if ma > mb else (b, a)
                weight = 1.0
                if phase == 'uncertainty':
                    def se(row):
                        values = row['ratings']
                        return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
                    gap = abs(ma - mb)
                    weight = float(gap / (gap + se(a) + se(b) + 1e-6))
                pairs.append((name, better['caption'], worse['caption'], weight))
    random.Random(seed).shuffle(pairs)
    return pairs[:cfg['human_pair_cap']]


def fit(cache, train_rows, validation_rows, cfg, hyper, seed, phase, device,
        baseline_validation=None, epochs=None, ablation=None, scale=2.5, clamp=True):
    seed_all(seed)
    head = BoundedAdapter(cache['image'].shape[1], cfg['rank'], cfg['radius']).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=hyper['learning_rate'], weight_decay=hyper['weight_decay'])
    domains = ['sugar'] if ablation == 'sugar_only' else ['flickr', 'polaris'] if ablation == 'human_only' else ['sugar', 'flickr', 'polaris']
    tables = {d: pair_table(train_rows, d, hyper, seed, phase) for d in domains}
    if any(not table for table in tables.values()):
        raise ValueError('No ranking supervision for one training domain')
    arrays = {}
    for d, table in tables.items():
        arrays[d] = [torch.tensor(x, device=device, dtype=torch.float32) for x in [
            np.asarray([cache['image'][cache['image_index'][r[0]]] for r in table]),
            np.asarray([cache['text'][cache['text_index'][r[1]]] for r in table]),
            np.asarray([cache['text'][cache['text_index'][r[2]]] for r in table]),
            np.asarray([r[3] for r in table])]]
    generator = torch.Generator(device=device).manual_seed(seed)
    history, best, best_state, best_epoch = [], -float('inf'), None, 1
    total_epochs = epochs or hyper['epochs']
    # Equal domain loss weight. Fixed steps from Sugar's training pair budget.
    n_sugar = sum(r['domain'] == 'sugar' for r in train_rows)
    steps = max(1, math.ceil(n_sugar / hyper['batch_size']))
    for epoch in range(1, total_epochs + 1):
        head.train()
        epoch_loss = 0.0
        for _ in range(steps):
            losses = []
            for d, (u, v, w, weights) in arrays.items():
                ids = torch.randint(len(u), (min(hyper['batch_size'], len(u)),), device=device, generator=generator)
                a, b, c = u[ids], v[ids], w[ids]
                ua, vb, wc = head(a), head(b), head(c)
                positive, negative = (ua * vb).sum(-1), (ua * wc).sum(-1)
                margin = hyper['sugar_margin'] if d == 'sugar' else 0.0
                rank_loss = (F.softplus((margin - positive + negative) / hyper['temperature']) * weights[ids]).sum() / weights[ids].sum().clamp_min(1e-8)
                preservation = ((ua - a).square().sum(-1) + (vb - b).square().sum(-1) + (wc - c).square().sum(-1)).mean()
                score_preservation = ((positive - (a*b).sum(-1)).square() + (negative - (a*c).sum(-1)).square()).mean()
                penalty = 0.0 if ablation == 'no_preservation' else cfg['preservation']
                losses.append(rank_loss + penalty * (preservation + score_preservation))
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
        head.eval()
        entry = {'epoch': epoch, 'loss': epoch_loss / steps}
        if validation_rows:
            pred, neg = prediction_pack(cache, validation_rows, head, device, scale, clamp)
            metrics = partition_metrics(validation_rows, pred, neg)
            gain = (validation_vector(metrics) - baseline_validation) / [0.01, 0.02, 0.02]
            objective = float(np.min(gain))
            entry.update(metrics=metrics, normalized_gains=gain.tolist(), objective=objective)
            if objective > best:
                best, best_state, best_epoch = objective, copy.deepcopy(head.state_dict()), epoch
        history.append(entry)
    if validation_rows:
        head.load_state_dict(best_state)
    return head.eval(), history, best_epoch


def select_config(trials):
    eligible = [t for t in trials if min(t['mean_normalized_gain']) >= 0]
    if not eligible:
        return {'fallback_base': True, 'reason': 'No configuration passed all validation guards'}
    chosen = max(eligible, key=lambda t: (min(t['mean_normalized_gain']), -t['config']['rank'], -t['config']['radius']))
    return {'fallback_base': False, 'config': chosen['config'],
            'epochs': max(1, int(np.median(chosen['best_epochs']))),
            'mean_normalized_gain': chosen['mean_normalized_gain']}
