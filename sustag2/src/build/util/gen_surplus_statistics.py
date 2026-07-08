"""Build-only: compute global min/max surplus for the colormap; write the stats file
that access reads via _min_surplus/_max_surplus (access imports the FILE, not this module,
so this stays build-side). TODO: port from data/surplus/gen_surplus_statistics.py."""
