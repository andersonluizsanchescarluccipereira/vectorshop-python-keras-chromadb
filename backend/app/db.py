import sqlite3
from pathlib import Path
from .seed import PRODUCTS, USERS, PURCHASES

class Catalog:
    def __init__(self, path):
        if path != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=15)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys = ON')
        self.db.execute('PRAGMA journal_mode = WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL, color TEXT NOT NULL, price REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS purchases (
          user_id INTEGER NOT NULL REFERENCES users(id),
          product_id INTEGER NOT NULL REFERENCES products(id),
          purchased_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY(user_id,product_id)
        );
        ''')
        if not self.db.execute('SELECT 1 FROM users LIMIT 1').fetchone():
            with self.db:
                self.db.executemany('INSERT INTO products VALUES (?,?,?,?,?)',PRODUCTS)
                self.db.executemany('INSERT INTO users VALUES (?,?,?)',USERS)
                self.db.executemany('INSERT INTO purchases (user_id,product_id) VALUES (?,?)',PURCHASES)
    def query(self,sql,args=()):
        return [dict(row) for row in self.db.execute(sql,args)]
    def users(self):
        return self.query('SELECT * FROM users ORDER BY CASE WHEN id=99 THEN 0 ELSE 1 END,id')
    def products(self):
        return self.query('SELECT * FROM products ORDER BY id')
    def user(self,uid):
        return next(iter(self.query('SELECT * FROM users WHERE id=?',(uid,))),None)
    def product(self,pid):
        return next(iter(self.query('SELECT * FROM products WHERE id=?',(pid,))),None)
    def purchases(self,uid):
        return self.query('''SELECT p.* FROM purchases pu JOIN products p ON p.id=pu.product_id
            WHERE pu.user_id=? ORDER BY pu.purchased_at DESC,p.id ASC''',(uid,))
    def add_purchase(self,uid,pid):
        with self.db:
            cur=self.db.execute('INSERT OR IGNORE INTO purchases (user_id,product_id) VALUES (?,?)',(uid,pid))
        return cur.rowcount>0
    def remove_purchase(self,uid,pid):
        with self.db:
            cur=self.db.execute('DELETE FROM purchases WHERE user_id=? AND product_id=?',(uid,pid))
        return cur.rowcount>0
    def popular(self,limit):
        return self.query('''SELECT p.*, COUNT(pu.product_id) AS purchase_count FROM products p
           LEFT JOIN purchases pu ON p.id=pu.product_id GROUP BY p.id
           ORDER BY purchase_count DESC,p.id LIMIT ?''',(limit,))
