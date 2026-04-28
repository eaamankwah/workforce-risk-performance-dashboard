from .query_base import QueryBase
from .sql_execution import QueryMixin


class Employee(QueryBase):

    name = "employee"

    def names(self):
        """
        Return a list of tuples containing (full_name, employee_id)
        for all employees.
        """
        sql = """
            SELECT first_name || ' ' || last_name AS full_name,
                   employee_id
            FROM employee
        """
        return self.query(sql)

    def username(self, id):
        """
        Return a list of tuples with the full name of the employee
        matching the given id.
        """
        sql = f"""
            SELECT first_name || ' ' || last_name AS full_name
            FROM employee
            WHERE employee_id = {id}
        """
        return self.query(sql)

    def model_data(self, id):
        """Return a pandas DataFrame with aggregated event data for ML model."""
        sql = f"""
            SELECT SUM(positive_events) positive_events,
                   SUM(negative_events) negative_events
            FROM {self.name}
            JOIN employee_events
                USING({self.name}_id)
            WHERE {self.name}.{self.name}_id = {id}
        """
        return self.pandas_query(sql)
