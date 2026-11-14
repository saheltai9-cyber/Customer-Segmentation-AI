# Customer Intelligence AI — Design System & Visual Specification

A unified design specification for the Customer Segmentation & Analytics SaaS Platform.

---

## 1. Design Vision & Philosophy

Designed with the aesthetic precision of modern enterprise analytics platforms (Stripe Dashboard, Linear, Raycast). The interface prioritizes clarity, high data density without clutter, restrained micro-interactions, and visual elegance.

- **Primary Goal:** Transform complex multivariate clustering models into actionable business intelligence.
- **Aesthetic:** Crisp light-mode enterprise UI featuring deep navy typography, energetic teal highlights, subtle slate borders, and generous whitespace.
- **Micro-Interactions:** Subtle card elevations on hover, smooth navigation transitions, clean tooltips on chart elements.

---

## 2. Color Palette & Tokens

| Token | Hex Value | Usage Role |
| :--- | :--- | :--- |
| `PRIMARY_NAVY` | `#172B4D` | Core brand identity, primary headings, centroid indicators |
| `ACCENT_TEAL` | `#0D9488` | Active navigation pill, primary action buttons, key metrics |
| `TEAL_LIGHT` | `#F0FDFA` | KPI icon background badge, pill badges, subtle card fills |
| `BG_LIGHT` | `#F5F7FA` | Full app body background |
| `CARD_BG` | `#FFFFFF` | Individual component cards, chart containers, sidebar |
| `TEXT_DARK` | `#182230` | Main body text, axis titles, table values |
| `TEXT_MUTED` | `#697586` | Secondary labels, subtexts, chart legends |
| `BORDER_COLOR` | `#E5EAF0` | Subtle card borders, dividers, table gridlines |
| `BORDER_HOVER` | `#CBD5E1` | Card hover state border |

### Cluster Cohort Palette
Six distinct, accessible colors tailored for categorical chart distinction:
- **Cluster 0:** `#0D9488` (Teal)
- **Cluster 1:** `#172B4D` (Navy)
- **Cluster 2:** `#2563EB` (Royal Blue)
- **Cluster 3:** `#F59E0B` (Amber)
- **Cluster 4:** `#7C3AED` (Purple)
- **Cluster 5:** `#E11D48` (Rose)

---

## 3. Typography Hierarchy

- **Primary Heading Font:** `Plus Jakarta Sans`, sans-serif (700 / 600 weight, letter spacing -0.02em)
- **Body & Metrics Font:** `Inter`, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto (400 / 500 / 600 weight)
- **Data Display:** Tabular numbers for clean numeric alignment across cards and tables.

### Type Scale
- **H1 (Page Title):** 28px (1.75rem), Bold (700)
- **H2 (Section Header):** 20px (1.25rem), Semi-Bold (600)
- **KPI Value:** 30px (1.85rem), Bold (700), Line height 1.2
- **Body Regular:** 14px (0.875rem), Regular (400), Color `#182230`
- **Caption / Subtext:** 12px (0.75rem), Regular (400), Color `#697586`
- **Badges & Overlines:** 11px (0.7rem), Semi-Bold (600), Uppercase, Letter-spacing 0.05em

---

## 4. UI Components Specification

### 4.1. Navigation Sidebar
- Top brand mark: High-contrast `CI` monogram tile with uppercase title "CUSTOMER INTELLIGENCE" and subtitle "Segmentation & Analytics".
- Navigation Items: Integrated via `streamlit-option-menu` with Bootstrap/Lucide-style icons (`grid-1x2`, `pie-chart`, `people`, `person-badge`, `database`).
- Active State: Clean teal accent background with white text and smooth pill styling.
- Model Status Card: Anchored bottom status pill showing live model status (`K-Means k=6`, silhouette score, records verified).

### 4.2. Metric KPI Cards
- Structure: Clear label top left, rounded square icon container top right (`#F0FDFA`), bold prominent metric, descriptive contextual microcopy.
- Geometry: 12px border radius, 1px solid border (`#E5EAF0`), subtle shadow `0 1px 3px rgba(16,24,40,0.05)`.
- Hover Behavior: `-2px` vertical translation with soft elevation shadow.

### 4.3. Interactive Plotly Charts
- Styled with white backgrounds, muted gray grid lines (`#E5EAF0`), zero margins, and dark slate hover labels (`#172B4D`).
- Responsive auto-scaling with clean legend positioning outside plot area.

### 4.4. Cluster Persona Cards
- Left accent border (5px) matched to the cohort theme color.
- Data-derived dynamic labels determined from real income, spending, and age relative thresholds.
- Strategy and actionable recommendations for commercial targeting.

---

## 5. Responsive Behavior

- **Desktop (1440x900 & above):** 4-column KPI grid, dual-column chart containers, generous padding.
- **Laptop (1366x768):** Flexible fluid grid adapting cards without text truncation.
- **Mobile (390x844):** Single-column stacked cards, collapsible sidebar, horizontal scroll prevention on charts.

