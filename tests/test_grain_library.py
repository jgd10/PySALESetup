from PySALESetup.grain_library import GrainLibrary


class TestGrainLibrary:
    def test_init(self):
        g = GrainLibrary()
        assert len(g.grains) == 0

    def test_load_builtin(self):
        g = GrainLibrary()
        g.load_builtin_library()
        assert len(g.grains) > 0

    def test_random_grain(self):
        g = GrainLibrary()
        g.load_builtin_library()
        grain = g.random_grain()
        assert grain in g.grains
