# futbol-1x2 @ challenger

- URI: `models:/futbol-1x2@challenger`
- Versión: 1
- Estrategia: random_forest
- Endpoint: `POST http://127.0.0.1:5002/invocations`
- Respuesta: una fila por partido con las probabilidades `[local, empate, visitante]`.

## Esquema de entrada

```json
[
  {
    "type": "double",
    "name": "f_h_gf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_ga_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_sf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_sa_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_sotf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_sota_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_pts_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_gf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_ga_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_sf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_sa_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_sotf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_sota_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_pts_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_gf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_ga_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_sf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_sa_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_sotf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_sota_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_pts_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_gf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_ga_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_sf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_sa_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_sotf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_sota_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_home_pts_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_rest",
    "required": true
  },
  {
    "type": "double",
    "name": "f_h_n_prev",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_gf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_ga_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_sf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_sa_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_sotf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_sota_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_pts_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_gf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_ga_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_sf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_sa_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_sotf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_sota_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_pts_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_gf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_ga_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_sf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_sa_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_sotf_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_sota_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_pts_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_gf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_ga_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_sf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_sa_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_sotf_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_sota_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_away_pts_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_rest",
    "required": true
  },
  {
    "type": "double",
    "name": "f_a_n_prev",
    "required": true
  },
  {
    "type": "double",
    "name": "f_elo_h",
    "required": true
  },
  {
    "type": "double",
    "name": "f_elo_a",
    "required": true
  },
  {
    "type": "double",
    "name": "f_elo_diff",
    "required": true
  },
  {
    "type": "double",
    "name": "f_elo_exp_h",
    "required": true
  },
  {
    "type": "double",
    "name": "f_d_pts_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_d_gd_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_d_sot_l5",
    "required": true
  },
  {
    "type": "double",
    "name": "f_d_pts_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_d_gd_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_d_sot_l10",
    "required": true
  },
  {
    "type": "double",
    "name": "f_rest_diff",
    "required": true
  }
]
```

## Esquema de salida

```json
[
  {
    "type": "long",
    "required": true
  }
]
```

## Petición

```bash
curl http://127.0.0.1:5002/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["f_h_gf_l5", "f_h_ga_l5", "f_h_sf_l5", "f_h_sa_l5", "f_h_sotf_l5", "f_h_sota_l5", "f_h_pts_l5", "f_h_gf_l10", "f_h_ga_l10", "f_h_sf_l10", "f_h_sa_l10", "f_h_sotf_l10", "f_h_sota_l10", "f_h_pts_l10", "f_h_home_gf_l5", "f_h_home_ga_l5", "f_h_home_sf_l5", "f_h_home_sa_l5", "f_h_home_sotf_l5", "f_h_home_sota_l5", "f_h_home_pts_l5", "f_h_home_gf_l10", "f_h_home_ga_l10", "f_h_home_sf_l10", "f_h_home_sa_l10", "f_h_home_sotf_l10", "f_h_home_sota_l10", "f_h_home_pts_l10", "f_h_rest", "f_h_n_prev", "f_a_gf_l5", "f_a_ga_l5", "f_a_sf_l5", "f_a_sa_l5", "f_a_sotf_l5", "f_a_sota_l5", "f_a_pts_l5", "f_a_gf_l10", "f_a_ga_l10", "f_a_sf_l10", "f_a_sa_l10", "f_a_sotf_l10", "f_a_sota_l10", "f_a_pts_l10", "f_a_away_gf_l5", "f_a_away_ga_l5", "f_a_away_sf_l5", "f_a_away_sa_l5", "f_a_away_sotf_l5", "f_a_away_sota_l5", "f_a_away_pts_l5", "f_a_away_gf_l10", "f_a_away_ga_l10", "f_a_away_sf_l10", "f_a_away_sa_l10", "f_a_away_sotf_l10", "f_a_away_sota_l10", "f_a_away_pts_l10", "f_a_rest", "f_a_n_prev", "f_elo_h", "f_elo_a", "f_elo_diff", "f_elo_exp_h", "f_d_pts_l5", "f_d_gd_l5", "f_d_sot_l5", "f_d_pts_l10", "f_d_gd_l10", "f_d_sot_l10", "f_rest_diff"], "data": [[2.2, 2.0, 16.4, 14.6, 4.6, 6.8, 1.0, 2.0, 1.5, 18.4, 11.2, 5.5, 4.8, 1.7, 2.2, 1.0, 17.0, 9.4, 5.0, 3.8, 2.2, 2.4, 1.0, 17.0, 8.8, 5.0, 3.4, 2.4, 60.0, 228.0, 1.2, 1.2, 11.0, 11.8, 3.2, 3.6, 1.4, 1.1, 1.2, 13.4, 10.5, 3.9, 3.4, 1.2, 1.4, 1.6, 12.0, 10.2, 4.0, 3.2, 1.2, 1.9, 1.6, 13.5, 12.3, 5.4, 4.5, 1.4, 60.0, 152.0, 1656.888752738507, 1514.2557959221926, 142.6329568163144, 0.7625025677693411, -0.3999999999999999, 0.20000000000000018, -1.8000000000000003, 0.5, 0.5999999999999999, 0.20000000000000018, 0.0]]}}'
```

## Respuesta

```json
{
  "predictions": [
    [
      0.5885859747402894,
      0.24141051107648936,
      0.17000351418322113
    ]
  ]
}
```
