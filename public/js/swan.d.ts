export namespace SwanStorage {
    /**
     * Retrieves a file from OPFS storage, downloading it if not present.
     * @param {string} filename
     * @param {string} url
     * @returns {Promise<File>}
     */
    function getModelFile(filename: string, url: string): Promise<File>;
}
export namespace SwanPipeline {
    /**
     * Checks if the model is ready, and triggers installation if missing.
     * @param {typeof SwanUI} ui
     * @returns {Promise<void>}
     */
    function ensureReady(ui: any): Promise<void>;
}
export namespace SwanUI {
    let el: HTMLElement | null;
    /**
     * Logs a message to the terminal UI.
     * @param {string} msg
     * @param {'log' | 'error' | 'system'} type
     */
    function log(msg: string, type?: "log" | "error" | "system"): void;
}
