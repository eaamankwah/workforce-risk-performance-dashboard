from fasthtml.common import (
    Div,
    FastHTML,
    H1,
    H2,
    Link,
    Meta,
    P,
    Style,
    serve,
)
import matplotlib.pyplot as plt

from employee_events import Employee, Team
from utils import load_model
from base_components import (
    BaseComponent,
    DataTable,
    Dropdown,
    MatplotlibViz,
    Radio,
)
from combined_components import CombinedComponent, FormGroup


# Subclass of Dropdown called `ReportDropdown`
class ReportDropdown(Dropdown):

    def build_component(self, entity_id, model):
        # Set the label to the model's name attribute
        self.label = model.name
        # Return the parent class's build_component output
        return super().build_component(entity_id, model)

    def component_data(self, entity_id, model):
        # Call the model's names() method for the dropdown options
        return model.names()


# Subclass of BaseComponent called `Header`
class Header(BaseComponent):

    def build_component(self, entity_id, model):
        # Standout: Title dynamically updates based on entity type.
        # Examples include "Employee Performance" or "Team Performance".
        entity_type = model.name.title()
        return H1(f"{entity_type} Performance")


# Subclass of MatplotlibViz called `LineChart`
class LineChart(MatplotlibViz):

    def visualization(self, asset_id, model):
        # Get event counts data
        data = model.event_counts(asset_id)

        # Guard: render a placeholder if no event data exists for this entity
        fig, ax = plt.subplots()
        if data.empty:
            ax.text(
                0.5,
                0.5,
                'No event data available',
                ha='center',
                va='center',
                fontsize=14,
                color='grey',
                transform=ax.transAxes,
            )
            ax.set_title('Cumulative Event Counts')
            self.set_axis_styling(ax, bordercolor='black', fontcolor='black')
            return

        # Fill nulls with 0
        data = data.fillna(0)

        # Set event_date as index
        data = data.set_index('event_date')

        # Sort the index
        data = data.sort_index()

        # Convert to cumulative sums
        data = data.cumsum()

        # Rename columns
        data.columns = ['Positive', 'Negative']

        # Style chart
        ax.set_facecolor('#f7fafd')
        fig.patch.set_facecolor('#ffffff')
        data['Positive'].plot(
            ax=ax,
            color='#2e75b6',
            linewidth=2.5,
            linestyle='-',
            label='Positive',
        )
        data['Negative'].plot(
            ax=ax,
            color='#e74c3c',
            linewidth=2.5,
            linestyle='--',
            label='Negative',
        )
        ax.set_title(
            'Cumulative Event Counts',
            fontsize=13,
            fontweight='bold',
            color='#1a2f4e',
            pad=12,
        )
        ax.set_xlabel(
            'Date',
            fontsize=10,
            color='#445566',
            labelpad=8,
        )
        ax.set_ylabel(
            'Cumulative Count',
            fontsize=10,
            color='#445566',
            labelpad=8,
        )
        ax.tick_params(colors='#7a90a8', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#d0dde9')
        ax.grid(
            True,
            linestyle='--',
            linewidth=0.5,
            color='#d0dde9',
            alpha=0.7,
        )
        legend = ax.legend(fontsize=9, framealpha=0.9, edgecolor='#d0dde9')
        legend.get_frame().set_facecolor('#f7fafd')
        fig.tight_layout(pad=1.5)


# Subclass of MatplotlibViz called `BarChart`
class BarChart(MatplotlibViz):

    # Load ML model as class attribute
    predictor = load_model()

    # Compute population-wide risk range for a given entity type.
    # Called at render time so employee views use employee scores
    # and team views use team scores — each gets its own relative scale.
    @classmethod
    def _population_scores(cls, model):
        """Return sorted list of all risk scores for the same entity type."""
        scores = []
        if model.name == 'team':
            t = Team()
            for _, entity_id in t.names():
                d = t.model_data(entity_id)
                if not d.empty and not d.isnull().all().all():
                    score = cls.predictor.predict_proba(d)[:, 1].mean()
                    scores.append(float(score))
        else:
            e = Employee()
            for _, entity_id in e.names():
                d = e.model_data(entity_id)
                if not d.empty and not d.isnull().all().all():
                    scores.append(float(cls.predictor.predict_proba(d)[0][1]))
        return sorted(scores) if scores else [0.0, 1.0]

    def visualization(self, asset_id, model):
        import matplotlib.colors as mcolors
        import numpy as np

        # Get model data
        data = model.model_data(asset_id)

        # Guard: render a placeholder if no model data exists
        fig, ax = plt.subplots(figsize=(6, 2.2))
        if data.empty or data.isnull().all().all():
            ax.text(0.5, 0.5, 'No data available for risk prediction',
                    ha='center', va='center', fontsize=13, color='#7a90a8',
                    transform=ax.transAxes)
            ax.set_title('Predicted Recruitment Risk', fontsize=13,
                         fontweight='bold', color='#1a2f4e', pad=10)
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f7fafd')
            return

        # Get prediction probabilities
        probas = self.predictor.predict_proba(data)[:, 1]

        # Employee = single value; Team = mean across members
        pred = float(probas.mean() if model.name == 'team' else probas[0])

        # ── Population scores for this entity type ─────────────────────────
        all_scores = self._population_scores(model)
        pop_min = all_scores[0]
        pop_max = all_scores[-1]
        padding = (pop_max - pop_min) * 0.15
        x_min = max(0.0, pop_min - padding)
        x_max = min(1.0, pop_max + padding)
        # Clamp axis to always include the current value
        x_min = min(x_min, pred * 0.85)
        x_max = max(x_max, pred * 1.15)

        # ── Colour scale: rank-based normalisation ───────────────────────────
        # Rank pred within the population so colour reflects relative standing,
        # not absolute distance — works correctly for any distribution shape
        n = len(all_scores)
        rank = sum(1 for s in all_scores if s < pred)
        norm_pred = float(rank / max(n - 1, 1))
        norm_pred = float(np.clip(norm_pred, 0.0, 1.0))

        cmap = mcolors.LinearSegmentedColormap.from_list(
            'risk', ['#2ecc71', '#f39c12', '#e74c3c']
        )
        bar_color = cmap(norm_pred)

        # ── Risk tier labels: rank-based thirds ──────────────────────────────
        # Rank-based assignment guarantees all three tiers are always populated
        # regardless of how tightly clustered the scores are.
        if norm_pred < 0.33:
            risk_label, risk_color = 'LOW RISK', '#27ae60'
        elif norm_pred < 0.67:
            risk_label, risk_color = 'MODERATE RISK', '#f39c12'
        else:
            risk_label, risk_color = 'HIGH RISK', '#e74c3c'

        # ── Draw bar ─────────────────────────────────────────────────────────
        ax.barh(
            [''],
            [pred],
            color=[bar_color],
            height=0.45,
            left=x_min,
            zorder=3,
        )
        # Background track
        ax.barh(
            [''],
            [x_max - x_min],
            color='#e8eff6',
            height=0.45,
            left=x_min,
            zorder=1,
        )
        ax.barh(
            [''],
            [pred - x_min],
            color=[bar_color],
            height=0.45,
            left=x_min,
            zorder=2,
        )

        ax.set_xlim(x_min, x_max)

        # ── Risk label and percentage inside / beside bar ────────────────────
        ax.text(x_max + (x_max - x_min) * 0.02, 0,
                f'{pred:.1%}  {risk_label}',
                va='center', ha='left', fontsize=10.5,
                fontweight='bold', color=risk_color)

        # ── Population range reference lines ─────────────────────────────────
        ax.axvline(
            pop_min,
            color='#b0c4d8',
            linewidth=1,
            linestyle=':',
            zorder=4,
        )
        ax.axvline(
            pop_max,
            color='#b0c4d8',
            linewidth=1,
            linestyle=':',
            zorder=4,
        )
        ax.text(pop_min, 0.48, 'min', ha='center', va='bottom',
                fontsize=7.5, color='#a0b4c8')
        ax.text(pop_max, 0.48, 'max', ha='center', va='bottom',
                fontsize=7.5, color='#a0b4c8')

        # ── Colour gradient legend bar ─────────────────────────
        # ─────────────────────────────────────────────────────
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        legend_ax = fig.add_axes([0.125, 0.08, 0.6, 0.07])
        legend_ax.imshow(gradient, aspect='auto', cmap=cmap,
                         extent=[pop_min, pop_max, 0, 1])
        legend_ax.set_xlim(pop_min, pop_max)
        legend_ax.set_yticks([])
        legend_ax.tick_params(labelsize=7.5, colors='#7a90a8')
        for spine in legend_ax.spines.values():
            spine.set_edgecolor('#d0dde9')
        legend_ax.set_xlabel(
            'Risk Scale: Low → High (relative to workforce)',
            fontsize=8,
            color='#7a90a8',
            labelpad=4,
        )
        marker_x = pop_min + norm_pred * (pop_max - pop_min)
        legend_ax.axvline(marker_x, color='#1a2f4e', linewidth=2)

        # ── Chart styling ─────────────────────────
        # ─────────────────────────────────────
        ax.set_facecolor('#f7fafd')
        fig.patch.set_facecolor('#ffffff')
        ax.set_title(
            'Predicted Recruitment Risk',
            fontsize=13,
            fontweight='bold',
            color='#1a2f4e',
            pad=10,
        )
        ax.tick_params(colors='#7a90a8', labelsize=8.5)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.1%}')
        )
        for spine in ax.spines.values():
            spine.set_edgecolor('#d0dde9')
        ax.set_xlabel(
            'Probability of Recruitment',
            fontsize=9,
            color='#445566',
            labelpad=6,
        )
        fig.tight_layout(rect=[0, 0.2, 1.0, 1.0])


# Subclass of CombinedComponent called Visualizations
class Visualizations(CombinedComponent):

    children = [LineChart(), BarChart()]

    # Leave this line unchanged
    outer_div_type = Div(cls='grid')


# Subclass of DataTable called `NotesTable`
class NotesTable(DataTable):

    def component_data(self, entity_id, model):
        # Return notes for the entity
        return model.notes(entity_id)

    def build_component(self, entity_id, model):
        # Wrap table in a styled section with a heading
        table = super().build_component(entity_id, model)
        if table is None:
            return Div(
                H2(
                    "Manager Notes",
                    style=(
                        "font-size:1rem;font-weight:700;color:#1a2f4e;"
                        "margin-bottom:12px;padding-bottom:8px;"
                        "border-bottom:2px solid #2e75b6;"
                    ),
                ),
                P(
                    "No notes available for this selection.",
                    style=(
                        "color:#7a90a8;font-style:italic;"
                        "font-size:0.88rem;"
                    ),
                ),
                style=(
                    "background:#fff;border-radius:10px;"
                    "padding:24px 28px;"
                    "box-shadow:0 4px 16px rgba(30,50,80,.12);"
                ),
            )
        return Div(
            H2(
                "Manager Notes",
                style=(
                    "font-size:1rem;font-weight:700;color:#1a2f4e;"
                    "margin-bottom:16px;padding-bottom:8px;"
                    "border-bottom:2px solid #2e75b6;"
                ),
            ),
            table,
            style=(
                "background:#fff;border-radius:10px;"
                "padding:24px 28px;"
                "box-shadow:0 4px 16px rgba(30,50,80,.12);"
            ),
        )


class DashboardFilters(FormGroup):

    id = "top-filters"
    action = "/update_data"
    method = "POST"

    children = [
        Radio(
            values=["Employee", "Team"],
            name='profile_type',
            hx_get='/update_dropdown',
            hx_target='#selector',
        ),
        ReportDropdown(
            id="selector",
            name="user-selection",
        ),
    ]


# Subclass of CombinedComponent called `Report`
class Report(CombinedComponent):

    children = [
        Header(),
        DashboardFilters(),
        Visualizations(),
        NotesTable(),
    ]


# Initialize a fasthtml app
app = FastHTML(hdrs=[
    Meta(charset='utf-8'),
    Meta(name='viewport', content='width=device-width, initial-scale=1'),
    Link(rel='preconnect', href='https://fonts.googleapis.com'),
    Link(
        rel='stylesheet',
        href=(
            'https://fonts.googleapis.com/css2'
            '?family=Inter:wght@400;500;600;700&display=swap'
        ),
    ),
    Style(open('../assets/report.css').read()),
])

# Initialize the `Report` class
report = Report()


# Index route - defaults to Employee 1
@app.get('/')
def index():
    return report(1, Employee())


# Employee route
@app.get('/employee/{id}')
def employee_page(id: str):
    return report(id, Employee())


# Team route
@app.get('/team/{id}')
def team_page(id: str):
    return report(id, Team())


# Keep the below code unchanged!
@app.get('/update_dropdown{r}')
def update_dropdown(r):
    dropdown = DashboardFilters.children[1]
    print('PARAM', r.query_params['profile_type'])
    if r.query_params['profile_type'] == 'Team':
        return dropdown(None, Team())
    elif r.query_params['profile_type'] == 'Employee':
        return dropdown(None, Employee())


@app.post('/update_data')
async def update_data(r):
    from fasthtml.common import RedirectResponse
    data = await r.form()
    profile_type = data._dict['profile_type']
    id = data._dict['user-selection']
    if profile_type == 'Employee':
        return RedirectResponse(f"/employee/{id}", status_code=303)
    elif profile_type == 'Team':
        return RedirectResponse(f"/team/{id}", status_code=303)


serve()
