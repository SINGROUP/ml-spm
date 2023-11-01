import copy

import numpy as np

from ..utils import elements

from typing import Iterable, Tuple, Self


class Atom:
    """
    A class representing an atom with a position, element and a charge.

    Arguments:
        xyz: The xyz position of the atom.
        element: The element of the atom. Either atomic number or chemical symbol.
        q: The charge of the atom.
        classes: Classes for categorizing atom based on their chemical elements. Each class is a list of elements either
            as atomic numbers or as chemical symbols.
        class_weights: List of weights or one-hot vector for classes. The weights must sum to unity.

    Note: only one of **classes** and **class_weights** can be specified at the same time.
    """

    def __init__(
        self,
        xyz: Iterable[int],
        element: int | str = None,
        q: float = None,
        classes: Iterable[list[int | str]] = None,
        class_weights: Iterable[float] = None,
    ):
        self.xyz = list(xyz)

        if element is not None:
            if isinstance(element, str):
                try:
                    element = elements.index(element) + 1
                except ValueError:
                    raise ValueError(f"Invalid element {element} for atom.")
            self.element = element
        else:
            self.element = None

        if q is None:
            q = 0
        self.q = 0

        if classes is not None:
            assert class_weights is None, "Cannot have both classes and class_weights not be None."
            self.class_weights, self.class_index = self._get_class(classes)
        elif class_weights is not None:
            assert np.allclose(sum(class_weights), 1.0), "Class weights don't sum to unity."
            self.class_weights = list(class_weights)
            self.class_index = np.argmax(class_weights)
        else:
            self.class_weights = []
            self.class_index = None

    def _get_class(self, classes):
        cls_assign = [self.element in c for c in classes]
        try:
            ind = cls_assign.index(1)
        except ValueError:
            raise ValueError(f"Element {self.element} is not in any of the classes.")
        return list(np.eye(len(classes))[ind]), ind

    def copy(self) -> Self:
        """Return a deepcopy of this object"""
        return copy.deepcopy(self)

    def array(
        self, xyz: bool = False, q: bool = False, element: bool = False, class_index: bool = False, class_weights: bool = False
    ) -> np.ndarray:
        """
        Return an array representation of the atom in order [xyz, q, element, class_index, one_hot_class].

        Arguments:
            xyz: Include xyz coordinates.
            q: Include charge.
            element: Include element.
            class_index: Include class index.
            class_weights: Include class weights.

        Returns:
            Array with requested information.
        """
        arr = []
        if xyz:
            arr += self.xyz
        if q:
            arr += [self.q]
        if element:
            arr += [self.element]
        if class_index:
            arr += [self.class_index]
        if class_weights:
            arr += self.class_weights
        return np.array(arr)


class MoleculeGraph:
    """
    A class representing a molecule graph with atoms and bonds. The atoms are stored as a list of Atom objects.

    Arguments:
        atoms: Molecule atom position and elements. If an np.ndarray, then must be of shape ``(num_atoms, 4)``, where
            each row corresponds to one atom with ``[x, y, z, element]``.
        bonds: Indices of bonded atoms. Each bond is a tuple ``(bond_start, bond_end)`` with the indices of the atom that
            the bond connects.
        classes: Classes for categorizing atom based on their chemical elements. Each class is a list of elements either
            as atomic numbers or as chemical symbols.
        class_weights: List of weights or one-hot vector for classes. The weights must sum to unity.

    Note: only one of **classes** and **class_weights** can be specified at the same time.
    """

    def __init__(
        self,
        atoms: list[Atom] | np.ndarray,
        bonds: list[Tuple[int, int]],
        classes: Iterable[list[int | str]] = None,
        class_weights: Iterable[float] = None,
    ):
        if class_weights is not None:
            assert len(atoms) == len(class_weights), "The number of atoms and the number of class weights for atoms don't match"
        else:
            class_weights = [None] * len(atoms)
        self.atoms: list[Atom] = []
        for atom, cw in zip(atoms, class_weights):
            if isinstance(atom, Atom):
                self.atoms.append(atom)
            else:
                self.atoms.append(Atom(atom[:3], atom[-1], q=None, classes=classes, class_weights=cw))
        self.bonds = bonds

    def __len__(self):
        return len(self.atoms)

    def copy(self) -> Self:
        """Return a deepcopy of this object"""
        return copy.deepcopy(self)

    def remove_atoms(self, remove_inds: Iterable[int]) -> Tuple[Self, list[Tuple[Atom, list[int]]]]:
        """
        Remove atoms and corresponding bonds from a molecule graph.

        Arguments:
            remove_inds: Indices of atoms to remove.

        Returns:
            Tuple (**new_molecule**, **removed**), where

            - **new_molecule** - New molecule graph where the atoms and bonds have been removed.
            - **removed** - Removed atoms and bonds. Each list item is a tuple ``(atom, bonds)`` corresponding to one of the
              removed atoms. The bonds are encoded as an indicator list where 0 indicates no bond and 1 indicates a bond with
              the atom at the corresponding index in the new molecule.
        """

        remove_inds = np.array(remove_inds, dtype=int)
        assert (remove_inds < len(self.atoms)).all()

        # Remove atoms from molecule
        removed_atoms = [self.atoms[i] for i in remove_inds]
        new_atoms = [self.atoms[i] for i in range(len(self.atoms)) if i not in remove_inds]

        # Remove corresponding bonds from molecule
        removed_bonds = [[0] * len(new_atoms) for _ in range(len(remove_inds))]
        new_bonds = []

        for bond in self.bonds:
            bond0 = bond[0] - (remove_inds < bond[0]).sum()
            bond1 = bond[1] - (remove_inds < bond[1]).sum()

            if not (bond[0] in remove_inds or bond[1] in remove_inds):
                new_bonds.append((bond0, bond1))

            elif not (bond[0] in remove_inds and bond[1] in remove_inds):
                for i in range(len(remove_inds)):
                    if bond[0] == remove_inds[i]:
                        removed_bonds[i][bond1] = 1
                    elif bond[1] == remove_inds[i]:
                        removed_bonds[i][bond0] = 1

        new_molecule = MoleculeGraph(new_atoms, new_bonds)
        removed = [(atom, bonds) for atom, bonds in zip(removed_atoms, removed_bonds)]

        return new_molecule, removed

    def add_atom(self, atom: Atom, bonds: list[int]) -> Self:
        """
        Add an atom and bonds to molecule graph.

        Arguments:
            atom: Atom to add.
            bonds: Indicator list (0s and 1s) of bond connections from new atom to existing atoms in the graph.

        Returns:
            New molecule graph where the atom and bonds have been added.
        """
        n_atoms = len(self.atoms)
        new_atoms = self.atoms + [atom]
        new_bonds = self.bonds + [(i, n_atoms) for i, b in enumerate(bonds) if b == 1]
        new_molecule = MoleculeGraph(new_atoms, new_bonds)
        return new_molecule

    def permute(self, permutation: list[int]) -> Self:
        """
        Permute the index order of atoms and corresponding bond indices.

        Arguments:
            permutation: New index order. Has to be same length as the number of atoms in graph.

        Returns:
            New molecule graph with indices permuted.
        """
        if len(permutation) != len(self.atoms):
            raise ValueError(
                f"Length of permutation list {len(permutation)} does not match the number of atoms in graph {len(self.atoms)}"
            )
        new_atoms = [self.atoms[i].copy() for i in permutation]
        new_bonds = []
        for b in self.bonds:
            new_bonds.append((permutation.index(b[0]), permutation.index(b[1])))
        new_molecule = MoleculeGraph(new_atoms, new_bonds)
        return new_molecule

    def array(
        self, xyz: bool = False, q: bool = False, element: bool = False, class_index: bool = False, class_weights: bool = False
    ) -> np.ndarray | list:
        """
        Return an array representation of the atoms in the molecule in order [xyz, q, element, class_index, class_weights]

        Arguments:
            xyz: Include xyz coordinates.
            q: Include charge.
            element: Include element.
            class_index: Include class index.
            class_weights: Include class weights.

        Returns:
            Array with requested information. Each element in first dimension corresponds to one atom.
        """
        if len(self.atoms) > 0:
            arr = np.stack([atom.array(xyz, q, element, class_index, class_weights) for atom in self.atoms], axis=0)
        else:
            arr = []
        return arr

    def adjacency_matrix(self) -> np.ndarray:
        """
        Return the adjacency matrix of the graph.

        Returns:
            Adjacency matrix of shape ``(n_atoms, n_atoms)``, where the presence of bonds between pairs of atoms are
            indicated by binary values.

        """
        A = np.zeros((len(self.atoms), len(self.atoms)), dtype=int)
        bonds = np.array(self.bonds, dtype=int).T
        if len(bonds) > 0:
            b0, b1 = bonds[0], bonds[1]
            np.add.at(A, (b0, b1), 1)
            np.add.at(A, (b1, b0), 1)
        return A

    def transform_xy(
        self,
        shift: Tuple[float, float] = (0, 0),
        rot_xy: float = 0,
        flip_x: bool = False,
        flip_y: bool = False,
        center: Tuple[float, float] = (0, 0),
    ) -> Self:
        """
        Transform atom positions in the xy plane.

        Arguments:
            shift: Shift atom positions in xy plane. Performed before rotation and flip.
            rot_xy: Rotate atoms in xy plane by rot_xy degrees around center point.
            flip_x: Mirror atom positions in x direction with respect to the center point.
            flip_y: Mirror atom positions in y direction with respect to the center point.
            center: Point around which transformations are performed.

        Returns:
            A new molecule graph with rotated atom positions.
        """

        center = np.array(center)
        atom_pos = self.array(xyz=True)
        atom_pos[:, :2] += np.array(shift)

        if rot_xy:
            a = rot_xy / 180 * np.pi
            rot_mat = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            atom_pos[:, :2] -= center
            atom_pos[:, :2] = np.dot(atom_pos[:, :2], rot_mat.T)
            atom_pos[:, :2] += center

        if flip_x:
            atom_pos[:, 0] = 2 * center[0] - atom_pos[:, 0]

        if flip_y:
            atom_pos[:, 1] = 2 * center[1] - atom_pos[:, 1]

        new_atoms = []
        for atom, pos in zip(self.atoms, atom_pos):
            new_atom = copy.deepcopy(atom)
            new_atom.xyz = list(pos)
            new_atoms.append(new_atom)
        new_bonds = copy.deepcopy(self.bonds)

        return MoleculeGraph(new_atoms, new_bonds)

    def crop_atoms(self, box_borders: np.ndarray) -> Self:
        """
        Delete atoms that are outside of specified region.

        Arguments:
            box_borders: Real-space extent of the region outside of which atoms are deleted. The array should be of the form
                ``((x_start, y_start, z_start), (x_end, y_end, z_end))``.

        Returns:
            A new molecule graph without the deleted atoms.
        """

        remove_inds = []
        for i, atom in enumerate(self.atoms):
            pos = atom.array(xyz=True)
            if not (
                box_borders[0][0] <= pos[0] <= box_borders[1][0]
                and box_borders[0][1] <= pos[1] <= box_borders[1][1]
                and box_borders[0][2] <= pos[2] <= box_borders[1][2]
            ):
                remove_inds.append(i)
        new_molecule, _ = self.remove_atoms(remove_inds)

        return new_molecule

    def randomize_positions(self, sigma: Tuple[int, int, int] = (0.2, 0.2, 0.1)) -> Self:
        """
        Randomly displace atom positions according to a Gaussian distribution.

        Arguments:
            sigma: Standard deviations for displacement in x, y, and z directions in Ångstroms.

        Returns:
            New molecule graph with randomized atom positions.
        """
        new_mol = self.copy()
        if len(self) > 0:
            delta = np.random.normal(0.0, sigma, (len(self), 3))
            for i in range(len(self)):
                new_mol.atoms[i].xyz = list(delta[i] + self.atoms[i].xyz)
        return new_mol
