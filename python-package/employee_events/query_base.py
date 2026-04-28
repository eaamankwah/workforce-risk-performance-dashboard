from .sql_execution import QueryMixin


class QueryBase(QueryMixin):

    # Class attribute for the table name
    name = ""

    def names(self):
        """Return an empty list - subclasses override this."""
        return []

    def event_counts(self, id):
        """
        Return a pandas DataFrame grouped by event_date with summed
        positive and negative events for the given id.
        """
        sql = f"""
            SELECT event_date,
                   SUM(positive_events) AS positive_events,
                   SUM(negative_events) AS negative_events
            FROM {self.name}
            JOIN employee_events
                USING({self.name}_id)
            WHERE {self.name}.{self.name}_id = {id}
            GROUP BY event_date
            ORDER BY event_date
        """
        return self.pandas_query(sql)

    def notes(self, id):
        """
        Return a pandas DataFrame with note_date and note columns
        from the notes table for the given id.
        """
        sql = f"""
            SELECT note_date, note
            FROM notes
            JOIN {self.name}
                USING({self.name}_id)
            WHERE {self.name}.{self.name}_id = {id}
        """
        return self.pandas_query(sql)
