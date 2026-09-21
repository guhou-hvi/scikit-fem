import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfem import (Basis, ElementLineP1, ElementTriHermite, ElementTriP1DG,
                   ElementTriMorley, FacetBasis, MeshTri1DG, MeshTri2)
from skfem.element import BOUNDARY_ELEMENT_MAP
from skfem.models.poisson import laplace, mass


@pytest.mark.parametrize('periodic', [[0], [1], [0, 1]])
def test_hermite_independent_of_boundary_registration(monkeypatch, periodic):
    def assemble():
        mesh = MeshTri1DG.init_tensor(
            np.linspace(0., 1., 4), np.linspace(0., 1., 4), periodic=periodic)
        basis = Basis(mesh, ElementTriHermite())
        return [form.assemble(basis).toarray() for form in (mass, laplace)]

    expected = assemble()
    monkeypatch.setitem(BOUNDARY_ELEMENT_MAP, ElementTriP1DG, ElementLineP1)
    for actual, reference in zip(assemble(), expected):
        assert_allclose(actual, reference, rtol=1e-9, atol=1e-10)


def test_custom_periodic_hermite_edge_data():
    class CustomHermite(ElementTriHermite):
        def gdof(self, functions, data, index):
            assert np.isfinite(data['e']).all()
            assert np.isfinite(data['n']).all()
            return super().gdof(functions, data, index)

    mesh = MeshTri1DG.init_tensor(np.linspace(0., 1., 4),
                                  np.linspace(0., 1., 4), periodic=[0])
    expected = mass.assemble(Basis(mesh, ElementTriHermite())).toarray()
    actual = mass.assemble(Basis(mesh, CustomHermite())).toarray()
    assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def test_curved_boundary_normal_projection():
    class CaptureMorley(ElementTriMorley):
        def gdof(self, functions, data, index):
            self.normal_data = data['n']
            return super().gdof(functions, data, index)

    mesh = MeshTri2.init_circle(1)
    element = CaptureMorley()
    Basis(mesh, element)
    boundary = FacetBasis(mesh, mesh.elem(), intorder=0)
    vertices = mesh.mapping().F(element.refdom.p)
    for position, facet in enumerate(boundary.find):
        cell = boundary.tind[position]
        local = np.flatnonzero(mesh.t2f[:, cell] == facet)[0]
        endpoints = mesh.elem.refdom.facets[local]
        tangent = (vertices[:, cell, endpoints[1]]
                   - vertices[:, cell, endpoints[0]])
        normal = np.array([tangent[1], -tangent[0]])
        normal /= np.linalg.norm(normal)
        # Preserve the existing projection at the first quadrature point,
        # including its magnitude on a curved boundary.
        expected = normal * np.dot(normal, boundary.normals[:, position, 0])
        assert_allclose(element.normal_data[local, :, cell], expected,
                        rtol=1e-12, atol=1e-12)
