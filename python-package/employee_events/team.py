from .query_base import QueryBase
from .sql_execution import QueryMixin


class Team(QueryBase):

    name = "team"

    def names(self):
        """
        Return a list of tuples containing (team_name, team_id)
        for all teams.
        """
        sql = """
            SELECT team_name, team_id
            FROM team
        """
        return self.query(sql)

    def username(self, id):
        """
        Return a list of tuples with the team_name of the team
        matching the given id.
        """
        sql = f"""
            SELECT team_name
            FROM team
            WHERE team_id = {id}
        """
        return self.query(sql)

    def model_data(self, id):
        """Return a pandas DataFrame with per-employee event data for ML model."""
        sql = f"""
            SELECT positive_events, negative_events FROM (
                SELECT employee_id,
                       SUM(positive_events) positive_events,
                       SUM(negative_events) negative_events
                FROM {self.name}
                JOIN employee_events
                    USING({self.name}_id)
                WHERE {self.name}.{self.name}_id = {id}
                GROUP BY employee_id
            )
        """
        return self.pandas_query(sql)
