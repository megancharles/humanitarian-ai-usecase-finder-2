# Reports folder

Drop `.txt` files here - one per situation report. The pipeline reads every
`.txt` file in this folder.

**File format** (plain text, no markup):

```
<ISO3 country code>
<report URL>
<report text, any number of lines>
```

- **Line 1**: the three-letter ISO3 country code the report is about (e.g. `MOZ`, `SSD`, `HTI`). This is used for the dashboard's country filter and map.
- **Line 2**: the source URL for the report (e.g. its ReliefWeb page). This is shown as a link in the dashboard.
- **Line 3 onward**: the report text itself, in as much or as little detail as you have. Paste the raw text of a situation report, assessment, or appeal.

Example:

```
MOZ
https://reliefweb.int/report/mozambique/example-situation-report
Mozambique Situation Report - Cyclone Response
...
```

A sample file, `sample_report.txt`, is included so you can test the pipeline end to end before adding real reports.
