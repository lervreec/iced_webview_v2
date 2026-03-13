use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};

use iced::keyboard;
use iced::mouse::{self, Interaction};
use iced::{Point, Size};
use rand::Rng;

use super::{Engine, PageType, PixelFormat, ViewId};
use crate::ImageInfo;

use anyrender::render_to_buffer;
use anyrender_vello_cpu::VelloCpuImageRenderer;
use blitz_dom::{Document, DocumentConfig};
use blitz_html::HtmlDocument;
use blitz_net::Provider;
use blitz_paint::paint_scene;
use blitz_traits::events::{
    BlitzKeyEvent, BlitzPointerEvent, BlitzPointerId, KeyState, MouseEventButton,
    MouseEventButtons, PointerCoords, PointerDetails, UiEvent,
};
use blitz_traits::navigation::{NavigationOptions, NavigationProvider};
use blitz_traits::net::NetProvider;
use blitz_traits::shell::{ColorScheme, ShellProvider, Viewport};
use cursor_icon::CursorIcon;
use keyboard_types::Modifiers;
use smol_str::SmolStr;

/// Captures link clicks from the Blitz document.
struct LinkCapture(Arc<Mutex<Option<String>>>);

impl NavigationProvider for LinkCapture {
    fn navigate_to(&self, options: NavigationOptions) {
        *self.0.lock().unwrap() = Some(options.url.to_string());
    }
}

/// Shell provider that tracks cursor and redraw requests.
struct WebviewShell {
    cursor: Arc<Mutex<CursorIcon>>,
}

impl ShellProvider for WebviewShell {
    fn set_cursor(&self, icon: CursorIcon) {
        *self.cursor.lock().unwrap() = icon;
    }
}

struct BlitzView {
    id: ViewId,
    document: Option<HtmlDocument>,
    net_provider: Arc<dyn NetProvider>,
    nav_capture: Arc<Mutex<Option<String>>>,
    cursor_icon: Arc<Mutex<CursorIcon>>,
    url: String,
    title: String,
    cursor: Interaction,
    last_frame: ImageInfo,
    last_frame_hash: u64,
    needs_render: bool,
    /// Number of update ticks to keep draining resources after goto().
    /// blitz_net fetches sub-resources (images, CSS) asynchronously; we need
    /// to call resolve() periodically to pick them up. Once the budget runs
    /// out we stop polling (resolve is expensive for large documents).
    resource_ticks: u32,
    scroll_y: f32,
    content_height: f32,
    size: Size<u32>,
    scale: f32,
}

/// CPU-based HTML rendering engine backed by Blitz (Stylo + Taffy + Vello).
///
/// Supports modern CSS (flexbox, grid, Firefox CSS engine via Stylo),
/// but no JavaScript. Uses `anyrender_vello_cpu` for software rasterization.
pub struct Blitz {
    views: Vec<BlitzView>,
    scale_factor: f32,
    color_scheme: ColorScheme,
}

fn detect_color_scheme() -> ColorScheme {
    if let Ok(val) = std::env::var("ICED_WEBVIEW_COLOR_SCHEME") {
        return match val.to_lowercase().as_str() {
            "dark" => ColorScheme::Dark,
            _ => ColorScheme::Light,
        };
    }
    if let Ok(theme) = std::env::var("GTK_THEME") {
        if theme.to_lowercase().contains("dark") {
            return ColorScheme::Dark;
        }
    }
    ColorScheme::Light
}

impl Default for Blitz {
    fn default() -> Self {
        Self {
            views: Vec::new(),
            scale_factor: 1.0,
            color_scheme: detect_color_scheme(),
        }
    }
}

impl Blitz {
    fn find_view(&self, id: ViewId) -> Option<&BlitzView> {
        self.views.iter().find(|v| v.id == id)
    }

    fn find_view_mut(&mut self, id: ViewId) -> Option<&mut BlitzView> {
        self.views.iter_mut().find(|v| v.id == id)
    }
}

fn cursor_icon_to_interaction(icon: CursorIcon) -> Interaction {
    match icon {
        CursorIcon::Pointer => Interaction::Pointer,
        CursorIcon::Text => Interaction::Text,
        CursorIcon::Crosshair => Interaction::Crosshair,
        CursorIcon::Grab => Interaction::Grab,
        CursorIcon::Grabbing => Interaction::Grabbing,
        CursorIcon::NotAllowed | CursorIcon::NoDrop => Interaction::NotAllowed,
        CursorIcon::ColResize | CursorIcon::EwResize => Interaction::ResizingHorizontally,
        CursorIcon::RowResize | CursorIcon::NsResize => Interaction::ResizingVertically,
        CursorIcon::ZoomIn => Interaction::ZoomIn,
        CursorIcon::ZoomOut => Interaction::ZoomOut,
        CursorIcon::Wait | CursorIcon::Progress => Interaction::Idle,
        _ => Interaction::Idle,
    }
}

/// Create a new net provider for sub-resource fetching.
fn new_net_provider() -> Arc<dyn NetProvider> {
    Provider::shared(None)
}

/// Parse HTML into a Blitz document with the given configuration.
fn create_document(
    html: &str,
    base_url: &str,
    net: &Arc<dyn NetProvider>,
    nav: &Arc<LinkCapture>,
    shell: &Arc<WebviewShell>,
    size: Size<u32>,
    scale: f32,
    color_scheme: ColorScheme,
) -> HtmlDocument {
    let phys_w = (size.width as f32 * scale) as u32;
    let phys_h = (size.height as f32 * scale) as u32;

    let config = DocumentConfig {
        base_url: if base_url.is_empty() {
            None
        } else {
            Some(base_url.to_string())
        },
        net_provider: Some(Arc::clone(net)),
        navigation_provider: Some(Arc::clone(nav) as Arc<dyn NavigationProvider>),
        shell_provider: Some(Arc::clone(shell) as Arc<dyn ShellProvider>),
        viewport: Some(Viewport::new(phys_w, phys_h, scale, color_scheme)),
        ..Default::default()
    };

    let mut doc = HtmlDocument::from_html(html, config);
    doc.resolve(0.0);
    doc
}

/// Max render height in logical pixels. Prevents multi-hundred-MB pixel
/// buffers for very tall documents (e.g. docs.rs pages). Content beyond
/// this height is reachable via scrolling but not pre-rasterized.
const MAX_RENDER_HEIGHT: f32 = 8192.0;

/// Render the document to an RGBA pixel buffer.
///
/// The buffer height is capped at `MAX_RENDER_HEIGHT` logical pixels to
/// keep memory and CPU usage bounded. The widget layer uses `content_height`
/// / `scroll_y` for scroll calculations; `content_height` is clamped to the
/// rendered height so the scrollbar range matches what's actually rasterized.
fn render_view(view: &mut BlitzView) {
    let w = view.size.width;
    let h = view.size.height;

    if w == 0 || h == 0 {
        return;
    }

    let doc = match view.document.as_ref() {
        Some(d) => d,
        None => {
            view.last_frame = ImageInfo::blank(w, h);
            view.needs_render = false;
            return;
        }
    };

    let root_height = doc.root_element().final_layout.size.height;
    let capped_height = root_height.min(MAX_RENDER_HEIGHT);
    view.content_height = capped_height;

    let scale = view.scale as f64;
    let render_w = (w as f64 * scale) as u32;
    let render_h = ((capped_height as f64).max(h as f64) * scale) as u32;

    if render_w == 0 || render_h == 0 {
        view.last_frame = ImageInfo::blank(w, h);
        view.needs_render = false;
        return;
    }

    let buffer = render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| {
            paint_scene(scene, doc, scale, render_w, render_h, 0, 0);
        },
        render_w,
        render_h,
    );

    let mut hasher = DefaultHasher::new();
    buffer.hash(&mut hasher);
    let new_hash = hasher.finish();

    if view.last_frame.image_width() == render_w
        && view.last_frame.image_height() == render_h
        && view.last_frame_hash == new_hash
    {
        view.needs_render = false;
        return;
    }

    view.last_frame = ImageInfo::new(buffer, PixelFormat::Rgba, render_w, render_h);
    view.last_frame_hash = new_hash;
    view.needs_render = false;
}

/// How many update ticks to keep draining resources after goto().
/// At 10ms per tick this gives ~30s for sub-resources to arrive.
const RESOURCE_TICK_BUDGET: u32 = 3000;

/// How often (in ticks) to actually call resolve() during the drain phase.
/// resolve() is expensive (full Stylo + Taffy layout pass), so we throttle it.
/// At ~10ms per tick, 100 ticks ≈ 1 second between resolve calls.
const RESOLVE_INTERVAL: u32 = 100;

/// Drain completed resource fetches and re-resolve the document.
fn drain_and_resolve(view: &mut BlitzView) {
    if let Some(ref mut doc) = view.document {
        doc.resolve(0.0);
    }
}

impl Engine for Blitz {
    /// Blitz cannot fetch the initial HTML page from a URL — the widget layer
    /// handles that via `fetch_html`. However, all sub-resource fetching
    /// (images, CSS `@import`) is handled internally by `blitz_net::Provider`,
    /// so the widget layer's image pipeline (`take_pending_images`,
    /// `load_image_from_bytes`) is not used. Returning `false` here is correct
    /// for its intended purpose: telling the widget layer to fetch page HTML.
    fn handles_urls(&self) -> bool {
        false
    }

    fn update(&mut self) {
        for view in &mut self.views {
            if view.resource_ticks > 0 {
                view.resource_ticks -= 1;
                if view.resource_ticks % RESOLVE_INTERVAL == 0 {
                    drain_and_resolve(view);
                    view.needs_render = true;
                }
            }
        }
    }

    fn render(&mut self, _size: Size<u32>) {
        for view in &mut self.views {
            if view.needs_render {
                render_view(view);
            }
        }
    }

    fn request_render(&mut self, id: ViewId, _size: Size<u32>) {
        let Some(view) = self.find_view_mut(id) else {
            return;
        };
        if view.needs_render {
            render_view(view);
        }
    }

    fn new_view(&mut self, size: Size<u32>, content: Option<PageType>) -> ViewId {
        let id = rand::thread_rng().gen();
        let w = size.width.max(1);
        let h = size.height.max(1);
        let size = Size::new(w, h);

        let nav_capture = Arc::new(Mutex::new(None));
        let cursor_icon = Arc::new(Mutex::new(CursorIcon::Default));
        let net = new_net_provider();
        let nav = Arc::new(LinkCapture(Arc::clone(&nav_capture)));
        let shell = Arc::new(WebviewShell {
            cursor: Arc::clone(&cursor_icon),
        });

        let (html, url) = match &content {
            Some(PageType::Html(html)) => (html.clone(), String::new()),
            Some(PageType::Url(url)) => (String::new(), url.clone()),
            None => (String::new(), String::new()),
        };

        let document = if !html.is_empty() {
            Some(create_document(
                &html,
                &url,
                &net,
                &nav,
                &shell,
                size,
                self.scale_factor,
                self.color_scheme,
            ))
        } else {
            None
        };
        let has_document = document.is_some();

        let mut view = BlitzView {
            id,
            document,
            net_provider: net,
            nav_capture,
            cursor_icon,
            url,
            title: String::new(),
            cursor: Interaction::Idle,
            last_frame: ImageInfo::blank(w, h),
            last_frame_hash: 0,
            needs_render: true,
            resource_ticks: if has_document {
                RESOURCE_TICK_BUDGET
            } else {
                0
            },
            scroll_y: 0.0,
            content_height: 0.0,
            size,
            scale: self.scale_factor,
        };

        render_view(&mut view);
        self.views.push(view);
        id
    }

    fn remove_view(&mut self, id: ViewId) {
        self.views.retain(|v| v.id != id);
    }

    fn has_view(&self, id: ViewId) -> bool {
        self.views.iter().any(|v| v.id == id)
    }

    fn view_ids(&self) -> Vec<ViewId> {
        self.views.iter().map(|v| v.id).collect()
    }

    fn focus(&mut self) {}

    fn unfocus(&self) {}

    fn resize(&mut self, size: Size<u32>) {
        for view in &mut self.views {
            view.size = size;
            if let Some(ref mut doc) = view.document {
                let scale = view.scale;
                let phys_w = (size.width as f32 * scale) as u32;
                let phys_h = (size.height as f32 * scale) as u32;
                let mut vp = doc.viewport_mut();
                vp.window_size = (phys_w, phys_h);
                drop(vp);
                doc.resolve(0.0);
            }
            view.needs_render = true;
        }
    }

    fn set_scale_factor(&mut self, scale: f32) {
        if (self.scale_factor - scale).abs() < f32::EPSILON {
            return;
        }
        self.scale_factor = scale;
        for view in &mut self.views {
            view.scale = scale;
            if let Some(ref mut doc) = view.document {
                let phys_w = (view.size.width as f32 * scale) as u32;
                let phys_h = (view.size.height as f32 * scale) as u32;
                let mut vp = doc.viewport_mut();
                vp.window_size = (phys_w, phys_h);
                vp.set_hidpi_scale(scale);
                drop(vp);
                doc.resolve(0.0);
            }
            view.needs_render = true;
        }
    }

    fn handle_keyboard_event(&mut self, id: ViewId, event: keyboard::Event) {
        let Some(view) = self.find_view_mut(id) else {
            return;
        };
        if let Some(ref mut doc) = view.document {
            if let Some(ke) = iced_keyboard_to_blitz(event) {
                let ui_event = if ke.state == KeyState::Pressed {
                    UiEvent::KeyDown(ke)
                } else {
                    UiEvent::KeyUp(ke)
                };
                doc.handle_ui_event(ui_event);
            }
        }
    }

    fn handle_mouse_event(&mut self, id: ViewId, point: Point, event: mouse::Event) {
        match event {
            mouse::Event::WheelScrolled { delta } => {
                self.scroll(id, delta);
            }
            mouse::Event::ButtonPressed(btn) => {
                let (button, buttons) = match btn {
                    mouse::Button::Left => (MouseEventButton::Main, MouseEventButtons::Primary),
                    mouse::Button::Right => {
                        (MouseEventButton::Secondary, MouseEventButtons::Secondary)
                    }
                    mouse::Button::Middle => {
                        (MouseEventButton::Auxiliary, MouseEventButtons::Auxiliary)
                    }
                    mouse::Button::Back => (MouseEventButton::Fourth, MouseEventButtons::Fourth),
                    mouse::Button::Forward => (MouseEventButton::Fifth, MouseEventButtons::Fifth),
                    _ => return,
                };
                let Some(view) = self.find_view_mut(id) else {
                    return;
                };
                if let Some(ref mut doc) = view.document {
                    let doc_y = point.y + view.scroll_y;
                    doc.handle_ui_event(UiEvent::PointerDown(BlitzPointerEvent {
                        id: BlitzPointerId::Mouse,
                        is_primary: true,
                        coords: PointerCoords {
                            page_x: point.x,
                            page_y: doc_y,
                            screen_x: point.x,
                            screen_y: point.y,
                            client_x: point.x,
                            client_y: point.y,
                        },
                        button,
                        buttons,
                        mods: Modifiers::empty(),
                        details: PointerDetails::default(),
                    }));
                }
            }
            mouse::Event::CursorMoved { .. } => {
                let Some(view) = self.find_view_mut(id) else {
                    return;
                };
                if let Some(ref mut doc) = view.document {
                    let doc_y = point.y + view.scroll_y;
                    doc.set_hover_to(point.x, doc_y);
                }
                let doc_cursor = view.document.as_ref().and_then(|d| d.get_cursor());
                let shell_cursor = *view.cursor_icon.lock().unwrap();
                let icon = doc_cursor.unwrap_or(shell_cursor);
                view.cursor = cursor_icon_to_interaction(icon);
            }
            mouse::Event::ButtonReleased(btn) => {
                let button = match btn {
                    mouse::Button::Left => MouseEventButton::Main,
                    mouse::Button::Right => MouseEventButton::Secondary,
                    mouse::Button::Middle => MouseEventButton::Auxiliary,
                    mouse::Button::Back => MouseEventButton::Fourth,
                    mouse::Button::Forward => MouseEventButton::Fifth,
                    _ => return,
                };
                let Some(view) = self.find_view_mut(id) else {
                    return;
                };
                if let Some(ref mut doc) = view.document {
                    let doc_y = point.y + view.scroll_y;
                    doc.handle_ui_event(UiEvent::PointerUp(BlitzPointerEvent {
                        id: BlitzPointerId::Mouse,
                        is_primary: true,
                        coords: PointerCoords {
                            page_x: point.x,
                            page_y: doc_y,
                            screen_x: point.x,
                            screen_y: point.y,
                            client_x: point.x,
                            client_y: point.y,
                        },
                        button,
                        buttons: MouseEventButtons::None,
                        mods: Modifiers::empty(),
                        details: PointerDetails::default(),
                    }));
                }
            }
            mouse::Event::CursorLeft => {
                if let Some(view) = self.find_view_mut(id) {
                    view.cursor = Interaction::Idle;
                }
            }
            _ => {}
        }
    }

    fn scroll(&mut self, id: ViewId, delta: mouse::ScrollDelta) {
        let Some(view) = self.find_view_mut(id) else {
            return;
        };
        match delta {
            mouse::ScrollDelta::Lines { y, .. } => {
                view.scroll_y -= y * 40.0;
            }
            mouse::ScrollDelta::Pixels { y, .. } => {
                view.scroll_y -= y;
            }
        }
        let max_scroll = (view.content_height - view.size.height as f32).max(0.0);
        view.scroll_y = view.scroll_y.clamp(0.0, max_scroll);
    }

    fn goto(&mut self, id: ViewId, page_type: PageType) {
        let color_scheme = self.color_scheme;
        let Some(view) = self.find_view_mut(id) else {
            return;
        };
        match page_type {
            PageType::Html(html) => {
                let nav = Arc::new(LinkCapture(Arc::clone(&view.nav_capture)));
                let shell = Arc::new(WebviewShell {
                    cursor: Arc::clone(&view.cursor_icon),
                });
                let net = new_net_provider();
                view.net_provider = Arc::clone(&net);

                view.document = Some(create_document(
                    &html,
                    &view.url,
                    &net,
                    &nav,
                    &shell,
                    view.size,
                    view.scale,
                    color_scheme,
                ));
                view.scroll_y = 0.0;
                view.needs_render = true;
                view.resource_ticks = RESOURCE_TICK_BUDGET;
            }
            PageType::Url(url) => {
                view.url = url;
            }
        }
    }

    fn refresh(&mut self, id: ViewId) {
        let Some(view) = self.find_view_mut(id) else {
            return;
        };
        if let Some(ref mut doc) = view.document {
            doc.resolve(0.0);
        }
        view.needs_render = true;
    }

    fn go_forward(&mut self, _id: ViewId) {}

    fn go_back(&mut self, _id: ViewId) {}

    fn get_url(&self, id: ViewId) -> String {
        let Some(view) = self.find_view(id) else {
            return "about:blank".to_string();
        };
        if view.url.is_empty() {
            "about:blank".to_string()
        } else {
            view.url.clone()
        }
    }

    fn get_title(&self, id: ViewId) -> String {
        self.find_view(id)
            .map(|v| v.title.clone())
            .unwrap_or_default()
    }

    fn get_cursor(&self, id: ViewId) -> Interaction {
        self.find_view(id)
            .map(|v| v.cursor)
            .unwrap_or(Interaction::Idle)
    }

    fn get_view(&self, id: ViewId) -> &ImageInfo {
        static BLANK: std::sync::LazyLock<ImageInfo> = std::sync::LazyLock::new(ImageInfo::default);
        self.find_view(id).map(|v| &v.last_frame).unwrap_or(&BLANK)
    }

    fn get_scroll_y(&self, id: ViewId) -> f32 {
        self.find_view(id).map(|v| v.scroll_y).unwrap_or(0.0)
    }

    fn get_content_height(&self, id: ViewId) -> f32 {
        self.find_view(id).map(|v| v.content_height).unwrap_or(0.0)
    }

    fn scroll_to_fragment(&mut self, id: ViewId, fragment: &str) -> bool {
        let Some(view) = self.find_view_mut(id) else {
            return false;
        };
        let doc = match view.document.as_ref() {
            Some(d) => d,
            None => return false,
        };

        // Try #id first (fast HashMap lookup), then [name="fragment"] via CSS selector.
        let node_id = doc.get_element_by_id(fragment).or_else(|| {
            let quoted = fragment.replace('\\', "\\\\").replace('"', "\\\"");
            doc.query_selector(&format!("[name=\"{quoted}\"]"))
                .ok()
                .flatten()
        });

        if let Some(nid) = node_id {
            if let Some(node) = doc.get_node(nid) {
                let pos = node.absolute_position(0.0, 0.0);
                let max_scroll = (view.content_height - view.size.height as f32).max(0.0);
                view.scroll_y = pos.y.clamp(0.0, max_scroll);
                return true;
            }
        }

        false
    }

    fn take_anchor_click(&mut self, id: ViewId) -> Option<String> {
        self.find_view_mut(id)?.nav_capture.lock().unwrap().take()
    }
}

fn iced_keyboard_to_blitz(event: keyboard::Event) -> Option<BlitzKeyEvent> {
    use keyboard_types::{Code, Key, Location};

    let (state, iced_key, iced_mods) = match event {
        keyboard::Event::KeyPressed { key, modifiers, .. } => (KeyState::Pressed, key, modifiers),
        keyboard::Event::KeyReleased { key, modifiers, .. } => (KeyState::Released, key, modifiers),
        _ => return None,
    };

    let kt_key = iced_key_to_blitz_key(&iced_key)?;

    let mut mods = Modifiers::empty();
    if iced_mods.shift() {
        mods |= Modifiers::SHIFT;
    }
    if iced_mods.control() {
        mods |= Modifiers::CONTROL;
    }
    if iced_mods.alt() {
        mods |= Modifiers::ALT;
    }
    if iced_mods.logo() {
        mods |= Modifiers::META;
    }

    let text = if state == KeyState::Pressed {
        match &kt_key {
            Key::Character(s) => Some(SmolStr::new(s)),
            _ => None,
        }
    } else {
        None
    };

    Some(BlitzKeyEvent {
        key: kt_key,
        code: Code::Unidentified,
        modifiers: mods,
        location: Location::Standard,
        is_auto_repeating: false,
        is_composing: false,
        state,
        text,
    })
}

fn iced_key_to_blitz_key(key: &keyboard::Key) -> Option<keyboard_types::Key> {
    use keyboard::key::Named;

    match key {
        keyboard::Key::Character(s) => Some(keyboard_types::Key::Character(s.to_string())),
        keyboard::Key::Named(named) => {
            let k = match named {
                Named::Enter => keyboard_types::Key::Enter,
                Named::Tab => keyboard_types::Key::Tab,
                Named::Space => keyboard_types::Key::Character(" ".to_string()),
                Named::Backspace => keyboard_types::Key::Backspace,
                Named::Delete => keyboard_types::Key::Delete,
                Named::Escape => keyboard_types::Key::Escape,
                Named::Insert => keyboard_types::Key::Insert,
                Named::CapsLock => keyboard_types::Key::CapsLock,
                Named::NumLock => keyboard_types::Key::NumLock,
                Named::ScrollLock => keyboard_types::Key::ScrollLock,
                Named::Pause => keyboard_types::Key::Pause,
                Named::PrintScreen => keyboard_types::Key::PrintScreen,
                Named::ContextMenu => keyboard_types::Key::ContextMenu,
                Named::ArrowDown => keyboard_types::Key::ArrowDown,
                Named::ArrowLeft => keyboard_types::Key::ArrowLeft,
                Named::ArrowRight => keyboard_types::Key::ArrowRight,
                Named::ArrowUp => keyboard_types::Key::ArrowUp,
                Named::End => keyboard_types::Key::End,
                Named::Home => keyboard_types::Key::Home,
                Named::PageDown => keyboard_types::Key::PageDown,
                Named::PageUp => keyboard_types::Key::PageUp,
                Named::F1 => keyboard_types::Key::F1,
                Named::F2 => keyboard_types::Key::F2,
                Named::F3 => keyboard_types::Key::F3,
                Named::F4 => keyboard_types::Key::F4,
                Named::F5 => keyboard_types::Key::F5,
                Named::F6 => keyboard_types::Key::F6,
                Named::F7 => keyboard_types::Key::F7,
                Named::F8 => keyboard_types::Key::F8,
                Named::F9 => keyboard_types::Key::F9,
                Named::F10 => keyboard_types::Key::F10,
                Named::F11 => keyboard_types::Key::F11,
                Named::F12 => keyboard_types::Key::F12,
                _ => return None,
            };
            Some(k)
        }
        _ => None,
    }
}
