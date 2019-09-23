from pathlib import Path
import sqlite3
import os
import logging
from tqdm import tqdm
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__BUILD_DB__")
query_logger = logging.getLogger("__QUERY_DB__")

path = Path(".../Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt")


def store_contents(data_path, save_path):
    """Preprocess and store a tencent word2vec sqlite.

    Args:
        data_path: Path to input (word2vec.txt)
        save_path: Path to output sqlite db.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE tencent_w2v (word text PRIMARY KEY, vector text);")

    count = 0
    with open(data_path, 'r') as f:
        next(f)
        for pairs in tqdm(f):
            p_split = pairs.split()
            w = p_split[:-200]
            if len(w) == 1:
                w = w[0]
            else:
                continue
            v = np.array(p_split[-200:], np.float32).tostring()  # np.fromstring(string, np.float32)
            try:
                c.execute("INSERT INTO tencent_w2v VALUES (:w, :v)", {'w': w, 'v': v})
                count += 1
            except:  # todo: so crazy, the program will not be interrupted by CTRL+C
                continue
    logger.info('Read %d words.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()
    logger.info("Sql connection closed...")


def query_w2v_from_db(word, db_path='./data/w2v.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    result = c.execute("SELECT vector FROM tencent_w2v WHERE word=?", (word,)).fetchall()
    # result = c.execute("SELECT * FROM tencent_w2v").fetchall()
    conn.commit()
    conn.close()
    try:
        result = np.fromstring(result[0][0], np.float32)
    except IndexError:
        query_logger.info("%s is not included in this database." % word)
        result = None
    return result


if __name__ == "__main__":
    x = query_w2v_from_db("中华人民共和国")
    print()