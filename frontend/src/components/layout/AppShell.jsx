import { ChevronRight, HeartPulse, UserRound } from "lucide-react";
import { NavLink, Outlet, useLocation } from "react-router";
import { formatRole, getVisibleNavigationSections, navigationSections } from "../../config/navigation.js";

function getUserName(user) {
  return user?.full_name || user?.fullName || user?.name || user?.email || "MedFlow user";
}

function getInitials(name) {
  return name
    .split(/[\s@._-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part.charAt(0).toUpperCase())
    .join("");
}

function UserProfile({ user, onLogout }) {
  if (!user) {
    return (
      <div className="profile-chip" aria-label="No active user session">
        <span className="profile-avatar profile-avatar-muted">
          <UserRound size={17} aria-hidden="true" />
        </span>
        <span className="profile-copy">
          <strong>No active session</strong>
          <span>Profile unavailable</span>
        </span>
      </div>
    );
  }

  const name = getUserName(user);
  const role = formatRole(user.role);

  return (
    <details className="profile-menu">
      <summary className="profile-chip">
        <span className="profile-avatar">{getInitials(name) || <UserRound size={17} />}</span>
        <span className="profile-copy">
          <strong>{name}</strong>
          <span>{role}</span>
        </span>
        <ChevronRight size={15} className="profile-chevron" aria-hidden="true" />
      </summary>
      <div className="profile-popover">
        <span className="profile-popover-label">Signed in as</span>
        <strong>{name}</strong>
        <span className="profile-popover-email">{user.email}</span>
        <span>{role}</span>
        {onLogout ? (
          <button className="profile-logout" type="button" onClick={onLogout}>
            Sign out
          </button>
        ) : null}
      </div>
    </details>
  );
}

export default function AppShell({ user = null, onLogout }) {
  const { pathname } = useLocation();
  const sections = getVisibleNavigationSections(user);
  const currentItem = navigationSections
    .flatMap((section) => section.items.map((item) => ({ ...item, section: section.label })))
    .filter((item) => pathname === item.href || (item.href !== "/" && pathname.startsWith(`${item.href}/`)))
    .sort((first, second) => second.href.length - first.href.length)[0];
  const title = currentItem?.label ?? "Page not found";
  const group = currentItem?.section ?? "Workspace";

  return (
    <div className="app-frame">
      <aside className="sidebar">
        <div className="brand-lockup">
          <span className="brand-mark">
            <HeartPulse size={22} strokeWidth={2.1} aria-hidden="true" />
          </span>
          <span className="brand-text">
            <strong>MedFlow <span>AI</span></strong>
            <small>Clinical workspace</small>
          </span>
        </div>

        <div className="sidebar-divider" />

        <nav className="sidebar-nav" aria-label="Primary navigation">
          {sections.map((section) => (
            <div className="nav-section" key={section.label}>
              <h2>{section.label}</h2>
              <div className="nav-items">
                {section.items.map((item) => {
                  const Icon = item.icon;
                  return (
                    <NavLink
                      className={({ isActive }) => `nav-link${isActive ? " is-active" : ""}`}
                      end={item.href === "/"}
                      key={item.href}
                      to={item.href}
                    >
                      <Icon size={18} strokeWidth={1.8} aria-hidden="true" />
                      <span>{item.label}</span>
                      {item.ai ? <span className="nav-ai-dot" aria-label="AI module" /> : null}
                    </NavLink>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className="sidebar-footer-mark"><HeartPulse size={16} aria-hidden="true" /></span>
          <span>
            <strong>Care teams, coordinated</strong>
            <small>MedFlow AI</small>
          </span>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="breadcrumb" aria-label="Breadcrumb">
            <span>Workspace</span>
            <ChevronRight size={14} aria-hidden="true" />
            <strong>{title}</strong>
            {group !== "Main" ? <><span className="breadcrumb-separator">·</span><span>{group}</span></> : null}
          </div>
          <div className="topbar-right">
            <span className="topbar-context">
              <span>Clinical operations</span>
            </span>
            <UserProfile user={user} onLogout={onLogout} />
          </div>
        </header>

        <main className={`page-content${pathname === "/clinical-assistant" ? " is-clinical-assistant" : ""}`}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
