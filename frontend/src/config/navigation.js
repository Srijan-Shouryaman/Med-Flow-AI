import {
  Activity,
  Brain,
  Building2,
  ClipboardList,
  FileText,
  LayoutDashboard,
  RefreshCw,
  Stethoscope,
  Users,
  UsersRound,
  BarChart3,
} from "lucide-react";

export const navigationSections = [
  {
    label: "Main",
    items: [
      {
        label: "Dashboard",
        href: "/",
        icon: LayoutDashboard,
        description: "A starting point for clinical operations.",
      },
      {
        label: "Patients",
        href: "/patients",
        icon: Users,
        description: "Patient records and reference lookup.",
      },
      {
        label: "Reports",
        href: "/reports",
        icon: FileText,
        description: "Medical report intake and processing.",
      },
      {
        label: "Cases",
        href: "/cases",
        icon: ClipboardList,
        description: "The clinical case workflow.",
      },
    ],
  },
  {
    label: "AI workspace",
    items: [
      {
        label: "AI Diagnosis",
        href: "/diagnosis",
        icon: Brain,
        description: "Standalone image-based AI diagnosis tools.",
        ai: true,
      },
      {
        label: "Predictions",
        href: "/predictions",
        icon: Activity,
        description: "AI department predictions and history.",
        ai: true,
        roles: ["SUPER_ADMIN", "HOSPITAL_ADMIN", "DEPARTMENT_HEAD"],
      },
      {
        label: "Recommendations",
        href: "/recommendations",
        icon: Stethoscope,
        description: "Advisory team recommendations.",
        ai: true,
      },
      {
        label: "Assignments",
        href: "/assignments",
        icon: ClipboardList,
        description: "Human-approved case assignments.",
      },
      {
        label: "Update Case Status",
        href: "/case-status",
        icon: RefreshCw,
        description: "Advance a case through its permitted workflow status.",
        roles: ["SUPER_ADMIN", "HOSPITAL_ADMIN", "DEPARTMENT_HEAD", "DOCTOR"],
      },
    ],
  },
  {
    label: "Operations",
    items: [
      {
        label: "Departments",
        href: "/departments",
        icon: Building2,
        description: "Department directory and details.",
        roles: ["SUPER_ADMIN", "HOSPITAL_ADMIN"],
      },
      {
        label: "Clinical Teams",
        href: "/teams",
        icon: UsersRound,
        description: "Clinical team directory.",
        roles: ["SUPER_ADMIN", "HOSPITAL_ADMIN", "DEPARTMENT_HEAD"],
      },
      {
        label: "Team Performance",
        href: "/team-performance",
        icon: BarChart3,
        description: "Team performance information.",
        roles: ["DEPARTMENT_HEAD"],
      },
    ],
  },
];

export function getVisibleNavigationSections(user) {
  return navigationSections
    .map((section) => ({
      ...section,
      items: section.items.filter(
        (item) => !item.roles?.length || hasRoleAccess(user, item.roles),
      ),
    }))
    .filter((section) => section.items.length > 0);
}

export function hasRoleAccess(user, allowedRoles = []) {
  const role = normalizeRole(user?.role);
  if (role === "SUPER_ADMIN") return true;
  return allowedRoles.map(normalizeRole).includes(role);
}

export function normalizeRole(role) {
  return typeof role === "string" ? role.trim().toUpperCase() : "";
}

export function formatRole(role) {
  if (typeof role !== "string" || !role.trim()) return "Role unavailable";
  return role
    .trim()
    .toLowerCase()
    .split(/[_\s]+/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
