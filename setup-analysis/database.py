import sqlite3
from sqlite3 import Error
from typing import List, Callable, Any, Tuple
from functools import wraps
from tabulate import tabulate
from pathlib import Path


def db_connection(func: Callable) -> Callable:
    """
    Decorator for managing database connections and transactions.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        try:
            conn = sqlite3.connect(self.path)
            result = func(self, conn, *args, **kwargs)
            conn.commit()
            return result
        except Error as e:
            print(e)
        finally:
            conn.close()

    return wrapper


class Database:
    def __init__(self, db_path):
        self.path: str | Path = Path(db_path)
        self.verbose: bool = True

        if not self.path.exists():
            sqlite3.connect(self.path)

        if not all([t in self.tables for t in self.create_statements.keys()]):
            self.create_tables()

    @property
    def create_statements(self):
        statements = dict(
            combinations="""CREATE TABLE IF NOT EXISTS combinations (
                itemID INTEGER PRIMARY KEY AUTOINCREMENT,
                figure1 INTEGER NOT NULL,
                figure2 INTEGER NOT NULL,
                figure3 INTEGER NOT NULL,
                figure4 INTEGER NOT NULL,
                figure5 INTEGER NOT NULL,
                figure6 INTEGER NOT NULL,
                figure7 INTEGER NOT NULL,
                figure8 INTEGER NOT NULL,
                pattern VARCHAR(32) NOT NULL
            );""",
            sequences="""CREATE TABLE IF NOT EXISTS sequences (
                itemID INTEGER PRIMARY KEY AUTOINCREMENT,
                combinationID INTEGER NOT NULL,
                figure1 INTEGER NOT NULL,
                figure2 INTEGER NOT NULL,
                figure3 INTEGER NOT NULL,
                figure4 INTEGER NOT NULL,
                figure5 INTEGER NOT NULL,
                figure6 INTEGER NOT NULL,
                figure7 INTEGER NOT NULL,
                figure8 INTEGER NOT NULL,
                choice1 INTEGER NOT NULL,
                choice2 INTEGER NOT NULL,
                choice3 INTEGER NOT NULL,
                choice4 INTEGER NOT NULL,
                maskedImgIdx INTEGER NOT NULL,
                seq_order VARCHAR(32) NOT NULL,
                choice_order VARCHAR(8) NOT NULL,
                pattern VARCHAR(32) NOT NULL,
                FOREIGN KEY(combinationID) REFERENCES combinations(itemID)
            );""",
            results="""CREATE TABLE IF NOT EXISTS results (
                trial_id INTEGER PRIMARY KEY,
                problem_id INTEGER NOT NULL,
                trial_onset_time FLOAT NOT NULL,
                series_end_time FLOAT NOT NULL,
                choice_onset_time FLOAT NOT NULL,
                rt FLOAT NOT NULL,
                rt_global FLOAT NOT NULL,
                choice_key VARCHAR(1) NOT NULL,
                solution_key VARCHAR(1) NOT NULL,
                choice TEXT NOT NULL,
                solution TEXT NOT NULL,
                correct TEXT NOT NULL,
                stim_pres_times TEXT NOT NULL
            );""",
            results_pilot="""CREATE TABLE IF NOT EXISTS results_pilot (
                idx INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_idx INTEGER NOT NULL,
                participant_id INTEGER NOT NULL,
                trial_type VARCHAR(20) NOT NULL,
                item_id INTEGER NOT NULL,
                trial_onset_time FLOAT NOT NULL,
                series_end_time FLOAT NOT NULL,
                choice_onset_time FLOAT NOT NULL,
                rt FLOAT NOT NULL,
                rt_global FLOAT NOT NULL,
                choice_key VARCHAR(20) NOT NULL,
                solution_key VARCHAR(20) NOT NULL,
                choice VARCHAR(20) NOT NULL,
                solution VARCHAR(20) NOT NULL,
                correct VARCHAR(20) NOT NULL,
                pattern VARCHAR(20) NOT NULL,
                intertrial_time FLOAT NOT NULL
            );""",
        )

        return statements

    @property
    @db_connection
    def tables(self, conn):
        c = conn.cursor()
        result = c.execute("SELECT * FROM sqlite_master WHERE type='table';")
        headers = [i[0] for i in result.description]
        result = result.fetchall()
        result = sorted([i[2] for i in result])

        return result

    @db_connection
    def update_create_statements(self, conn):
        c = conn.cursor()
        create_stmts = {}
        try:
            for table in [t for t in self.tables if t != "sqlite_sequence"]:
                create_stmt = self.manage_table(table, "schema")
                create_stmt = create_stmt.replace(
                    "CREATE TABLE", "CREATE TABLE IF NOT EXISTS"
                )
                create_stmts[table] = create_stmt
        except Error as e:
            print(e)

        return create_stmts

    @db_connection
    def create_tables(self, conn):
        for tb, statement in self.create_statements.items():
            c = conn.cursor()
            print(statement) if self.verbose else None
            c.execute(statement)

    @db_connection
    def manage_table(self, conn, tb_name: str, action: str, lim: int = 10):
        commands = {
            "empty": f"DELETE FROM '{tb_name}';",
            "drop": f"DROP TABLE '{tb_name}';",
            "info": f"PRAGMA table_info('{tb_name}');",
            "head": f"SELECT * FROM '{tb_name}' LIMIT {lim};",
            "count": f"SELECT COUNT(*) FROM '{tb_name}';",
            "schema": f"SELECT sql \nFROM sqlite_schema \nWHERE name = '{tb_name}';",
        }
        valid_actions = [c.lower() for c in commands.keys()]

        if action.lower() not in valid_actions:
            raise ValueError(f"Invalid action. Must be one of {valid_actions}.")

        command = commands[action]
        c = conn.cursor()

        if action in ["info", "head", "count"]:
            results = c.execute(command)
            headers = [i[0] for i in results.description]
            results = tabulate(results.fetchall(), headers=headers)
            print(results)
        elif action == "schema":
            try:
                result = c.execute(command).fetchone()[0]
                print(result)
                return result
            except Error as e:
                print(e)
        elif action in ["empty", "drop"]:
            confirm = input(f"{action}, are you sure? y/n").lower()
            if confirm == "y":
                c.execute(command)
            else:
                print("operation canceled")

    @db_connection
    def insert_combinations(self, conn, combinations):
        if not combinations:
            raise ValueError(
                "The combinations list is empty. Please provide a list with at least one combination."
            )

        c = conn.cursor()

        # ! Assuming all combinations have the same length and structure
        tb_name = "combinations"
        tb_info = c.execute(f"PRAGMA table_info({tb_name});").fetchall()
        cols = [i[1] for i in tb_info]
        cols = cols[1:]  # * Because the first column is AUTOINCREMENT
        n_cols = len(cols)
        cols = ", ".join(cols)

        placeholder = ", ".join(["?"] * n_cols)

        command = f"INSERT INTO {tb_name} ({cols}) VALUES ({placeholder})"

        try:
            c.executemany(command, combinations)
        except Error as e:
            print(e)

    @db_connection
    def insert_sequences(self, conn, sequences):
        if not sequences:
            raise ValueError(
                "The sequences list is empty. Please provide a list with at least one sequence."
            )

        c = conn.cursor()

        # ! Assuming all combinations have the same length and structure
        tb_name = "sequences"
        tb_info = c.execute(f"PRAGMA table_info({tb_name});").fetchall()
        cols = [i[1] for i in tb_info]
        cols = cols[1:]  # * Because the first column is AUTOINCREMENT
        n_cols = len(cols)
        cols = ", ".join(cols)

        placeholder = ", ".join(["?"] * n_cols)

        command = f"INSERT INTO {tb_name} ({cols}) VALUES ({placeholder})"

        try:
            c.executemany(command, sequences)
        except Error as e:
            print(e)

    @db_connection
    def insert_results(self, conn, results):
        raise NotImplementedError

    @db_connection
    def insert_results_pilot(self, conn, results):
        if not results:
            raise ValueError(
                "The results list is empty. Please provide a list with at least one result."
            )

        c = conn.cursor()

        tb_name = "results_pilot"

        tb_info = c.execute(f"PRAGMA table_info({tb_name});").fetchall()

        cols = [i[1] for i in tb_info]
        cols = cols[1:]  # * Because the first column is AUTOINCREMENT

        n_cols = len(cols)
        cols = ", ".join(cols)

        placeholder = ", ".join(["?"] * n_cols)

        command = f"INSERT INTO {tb_name} ({cols}) VALUES ({placeholder})"

        try:
            c.executemany(command, results)
        except Error as e:
            print(f"Error inserting results into {tb_name} table: ", end="")
            raise

    @db_connection
    def find_duplicates(self, conn, tb_name: str, cols: List[str] = None):
        c = conn.cursor()
        if cols is None:
            tb_info = c.execute("PRAGMA table_info(combinations);").fetchall()
            cols = [i[1] for i in tb_info]

        cols = ", ".join(cols)

        command = f"""SELECT {cols}, COUNT(*)
        FROM {tb_name}
        GROUP BY {cols}
        HAVING COUNT(*) > 1"""

        results = c.execute(command).fetchall()
        print(f"{len(results)} duplicate(s) found.")

        return (i for i in results)

    @db_connection
    def execute(self, conn, query: str, fetch: bool = True):
        c = conn.cursor()
        try:
            c.execute(query)
            if fetch:
                return c.fetchall()
        except Error as e:
            print(e)


if __name__ == "__main__":
    wd = Path(__file__).parent
    db = Database(wd.parent / "config/database.db")
    db.manage_table("combinations", "head")
