import requests
import sys
from pathlib import Path
from abbyy_course_cvdl_t3 import coco_text
from abbyy_course_cvdl_t3.coco_text import COCO_Text
from abbyy_course_cvdl_t3.utils import evaluate_ap_from_cocotext_json


def update_leaderboard_score(username, score):
    base_url = 'https://keepthescore.co/api/{}/'
    rtoken = 'xcmgfzogolr'
    etoken = 'wodbohbooye'
    # читаем лидерборд
    r = requests.get(base_url.format(rtoken) + 'board')
    if not r.ok:
        return r.json()
    board = r.json()

    # находим текущий скор
    current_score = None
    for player_data in board['players']:
        if username == player_data['name']:
            current_score = player_data['score']
            current_id = player_data['id']
            break
    if current_score is None:
        return "username not found"

    # вычисляем поправку (api позволяет делать add, не set)
    delta = score - current_score
    if abs(delta) < 0.01: # на лб скор хранится с двумя знаками после запятой
        return

    # если есть разница - отправляем ее
    data = {"player_id": current_id, "score" : delta, 'comment': '{:+2.3f}'.format(delta)}
    r = requests.post(base_url.format(etoken) + 'score', json=data)
    if not r.ok:
        return r.json()


if __name__ == "__main__":
    base = Path(coco_text.__file__).absolute().parent
    gt_path = base / 'data' / 'cocotext.v2.json'
    assert gt_path.exists(), str(gt_path)
    pred_path = base.parent / 'predictions.json'
    assert pred_path.exists(), str(pred_path)
    ct = COCO_Text(gt_path)
    ap, prec, rec = evaluate_ap_from_cocotext_json(
        coco_text=ct,
        path=str(pred_path)
    )
    print("Итоговый скор AP на val: {:1.6f}".format(ap))
    if len(sys.argv) > 1:
        username = str(sys.argv[1])
        err_message = update_leaderboard_score(username, ap * 100)
        if err_message is None:
            print(f"Скор AP обновлен в таблице для пользователя '{username}'")
        else:
            print(f"Не удалось обновить скор в таблице для пользователя '{username}': {err_message}")
